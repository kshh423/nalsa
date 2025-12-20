import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from datetime import date, timedelta, datetime
import time
import pytz

# ==============================================================================
# 0. 전역 설정 및 상수 정의
# ==============================================================================
DEFAULT_BIG_TECH_TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA']
KST = pytz.timezone('Asia/Seoul')
NOW_KST = datetime.now(KST)
TODAY = NOW_KST.date()

# PER 기준 상수
PER_CRITERIA_DYNAMIC = {
    'BUY_3X': 30.0, 'BUY_2X': 32.0, 'BUY_1X': 35.0,
    'HOLD': 38.0, 'SELL_15': 41.0, 'SELL_30': 45.0, 'SELL_50': 45.0 # SELL_50은 SELL_30과 동일 기준값 사용
}
CASH_REINVESTMENT_RATIO = {'BUY_3X': 0.50, 'BUY_2X': 0.30, 'BUY_1X': 0.10}
SELL_RATIO = {'SELL_15': 0.15, 'SELL_30': 0.30, 'SELL_50': 0.50}

# PER 기준선 Plotly 스타일
PER_LINE_STYLES = {
    PER_CRITERIA_DYNAMIC['BUY_3X']: ('green', '30.0 (3X 매수)'),
    PER_CRITERIA_DYNAMIC['BUY_2X']: ('darkgreen', '32.0 (2X 매수)'),
    PER_CRITERIA_DYNAMIC['BUY_1X']: ('blue', '35.0 (1X 매수)'),
    PER_CRITERIA_DYNAMIC['HOLD']: ('orange', '38.0 (HOLD)'),
    PER_CRITERIA_DYNAMIC['SELL_15']: ('red', '41.0 (15% 매도)'),
    PER_CRITERIA_DYNAMIC['SELL_30']: ('darkred', '45.0 (30% 매도)')
}
PER_LEVELS_SORTED = sorted(list(set(PER_CRITERIA_DYNAMIC.values())))


# ==============================================================================
# 1. 데이터 로드 및 캐싱 함수
# ==============================================================================

@st.cache_data(ttl=3600)
def load_ticker_info(ticker, max_retries=3):
    """티커 정보를 로드합니다 (EPS, 회사 이름)."""
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker)
            info = data.info
            eps = info.get('trailingEps')
            if eps is None or eps == 0:
                eps = info.get('forwardEps')
            per_info = {
                'EPS': eps if eps else 0,
                'CompanyName': info.get('longName', ticker),
            }
            return per_info, None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
            else:
                return None, f"Ticker information could not be loaded after {max_retries} attempts: {e}"
    return None, "Unexpected failure in Ticker Info loading."

@st.cache_data(ttl=3600)
def load_historical_data(ticker, start_date, end_date, max_retries=3):
    """yfinance에서 주가 데이터를 로드합니다."""
    if start_date == 'max':
        start_date = None
    for attempt in range(max_retries):
        try:
            hist = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if hist.empty:
                return None, "해당 기간의 주가 데이터를 가져올 수 없습니다."
            return hist, None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
            else:
                return None, f"데이터 로드 중 오류가 발생했습니다: {e}"
    return None, "Unexpected failure in Historical Data loading."

@st.cache_data(ttl=3600)
def load_big_tech_data(tickers):
    """요청된 빅테크 종목의 최신 재무 정보를 로드합니다 (현재 PER 계산용)."""
    data_list = []
    
    # yfinance.Tickers를 사용하여 여러 티커에 대한 정보 요청을 효율적으로 처리
    tickers_obj = yf.Tickers(tickers)
    
    for ticker in tickers:
        try:
            info = tickers_obj.tickers[ticker].info
            market_cap = info.get('marketCap', np.nan)
            trailing_pe = info.get('trailingPE', np.nan)
            
            # 시가총액과 PER로 순이익 역산 (Net Income = Market Cap / Trailing PE)
            net_income = market_cap / trailing_pe if market_cap and trailing_pe and trailing_pe > 0 else np.nan
            
            data_list.append({
                'Ticker': ticker,
                'MarketCap': market_cap,
                'TrailingPE': trailing_pe,
                'NetIncome': net_income,
            })
        except Exception:
            # 실패한 경우 nan으로 처리
            data_list.append({'Ticker': ticker, 'MarketCap': np.nan, 'TrailingPE': np.nan, 'NetIncome': np.nan})
            
    return pd.DataFrame(data_list)
@st.cache_data(ttl=3600)
def calculate_accurate_group_per_history(ticker_list, start_date, end_date):
    """
    빅테크 그룹의 시가총액 가중 평균 PER의 정확한 역사적 시계열을 계산합니다.
    (yfinance.download로 주가 데이터를 병렬 로드하여 최적화)
    """
    
    start_date_yf = None
    end_date_yf = None
    period_arg = None
    
    if start_date == 'max':
        period_arg = 'max'
    else:
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        start_date_yf = start_date_dt.strftime('%Y-%m-%d')
        end_date_yf = end_date_dt.strftime('%Y-%m-%d')
        
    combined_market_cap = pd.DataFrame()
    combined_net_income = pd.DataFrame()
    valid_tickers = []
    
    with st.spinner("📊 PER 추이 계산 중... (yfinance 데이터 로드 및 최적화 적용)"):
        
        # 1. 주가 데이터 병렬 로드 (최적화 부분)
        try:
            hist_all, hist_error = load_historical_data(
                ticker_list, start_date=start_date_yf if start_date != 'max' else None, 
                end_date=end_date_yf, period=period_arg
            )
            if hist_all is None:
                return None, hist_error
            
            # 멀티 티커 로드 시 컬럼 이름이 (Adj Close, Ticker) 등으로 구성됨
            hist_closes = hist_all['Close'].dropna(axis=1, how='all')
            
        except Exception as e:
            return None, f"주가 데이터 병렬 로드 중 오류 발생: {e}"

        
        # 2. 개별 종목 정보 및 순이익 시계열 로드 (순차적 처리 필요)
        for ticker in ticker_list:
            if ticker not in hist_closes.columns: continue

            try:
                stock = yf.Ticker(ticker)
                
                # 주가 데이터 추출 및 인덱스 처리
                hist_close = hist_closes[ticker].dropna()
                if hist_close.empty: continue
                hist_close.index = hist_close.index.tz_localize(None)
                
                # 발행주식수 가져오기
                try:
                    shares = stock.fast_info['shares_outstanding']
                except:
                    shares = stock.info.get('sharesOutstanding')
                
                if not shares: continue

                # 일별 시가총액 계산
                combined_market_cap[ticker] = hist_close * shares
                
                # 순이익(Net Income) 데이터 가져오기
                income_stmt = stock.financials
                income_keys = ['Net Income', 'Net Income Common Stockholders']
                net_income_row = next((income_stmt.loc[k] for k in income_keys if k in income_stmt.index), None)
                
                if net_income_row is None: continue

                net_income_row.index = pd.to_datetime(net_income_row.index).tz_localize(None)
                net_income_row = net_income_row.sort_index()
                
                # 주가 날짜에 맞춰 순이익 데이터 확장 (다음 발표 전까지 유지)
                combined_net_income[ticker] = net_income_row.reindex(hist_close.index, method='ffill')
                valid_tickers.append(ticker)

            except Exception:
                continue

    if combined_market_cap.empty or combined_net_income.empty:
        return None, "유효한 Market Cap 및 Net Income 데이터를 가진 종목이 없어 PER 계산이 불가능합니다."

    # 데이터프레임 인덱스 정렬 및 동기화
    common_index = combined_market_cap.index.intersection(combined_net_income.index)
    
    # PER 계산 로직은 유지
    total_market_cap = combined_market_cap.loc[common_index, valid_tickers].sum(axis=1)
    total_net_income = combined_net_income.loc[common_index, valid_tickers].sum(axis=1)
    
    group_per = total_market_cap / total_net_income.mask(total_net_income <= 0)
    group_per = group_per.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    
    if group_per.empty:
        return None, "순이익이 양수인 기간의 데이터가 부족하여 그룹 PER 시계열을 계산할 수 없습니다."
        
    return group_per, None

@st.cache_data(ttl=3600)
def calculate_portfolio_metrics(ticker1, ticker2, start_date, end_date):
    """
    두 자산 포트폴리오의 효율적 투자선을 계산합니다.
    """
    tickers = [ticker1, ticker2]
    
    # 1. 주가 데이터 로드 (수익률 계산용)
    hist_data, error = load_historical_data(tickers, start_date, end_date)
    
    if error: return None, error, None
    
    # 2. 일별 수익률 계산
    returns = hist_data['Close'].pct_change().dropna()
    
    if returns.empty or len(returns) < 20: return None, "데이터 부족으로 수익률 계산 불가.", None
    
    # 3. 연간 환산 요소 (252 거래일)
    annual_factor = 252
    
    # 4. 연간 수익률 및 공분산 계산
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    
    # 5. 포트폴리오 시뮬레이션
    num_portfolios = 101 # 0%에서 100%까지 1% 단위로 시뮬레이션
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, num_portfolios)]
    
    portfolio_results = []
    
    for w in weights:
        # 포트폴리오 수익률: w1*R1 + w2*R2
        port_return = np.sum(mean_returns * w)
        
        # 포트폴리오 변동성: sqrt(wT * Cov * w)
        port_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        
        portfolio_results.append({
            'Return': port_return,
            'Volatility': port_volatility,
            'Weight_1': w[0],
            'Weight_2': w[1]
        })
        
    df_port = pd.DataFrame(portfolio_results)
    
    # 6. 주요 지점 계산 (MVP, Max Sharpe)
    # 샤프 비율 (무위험 이자율은 편의상 0으로 가정)
    df_port['Sharpe_Ratio'] = df_port['Return'] / df_port['Volatility']
    
    # 최소 분산 포트폴리오 (MVP)
    mvp = df_port.loc[df_port['Volatility'].idxmin()]
    
    # 최대 샤프 비율 포트폴리오
    max_sharpe = df_port.loc[df_port['Sharpe_Ratio'].idxmax()]
    asset_metrics = {
        ticker1: {'Return': mean_returns[ticker1], 'Volatility': returns[ticker1].std() * np.sqrt(annual_factor)},
        ticker2: {'Return': mean_returns[ticker2], 'Volatility': returns[ticker2].std() * np.sqrt(annual_factor)},
    }
    
    # 반환 구조 변경: asset_metrics 추가
    return df_port, None, {'mvp': mvp, 'max_sharpe': max_sharpe, 'asset_metrics': asset_metrics}


@st.cache_data(ttl=3600)
def calculate_multi_ticker_metrics(ticker_list, start_date, end_date):
    """여러 티커의 연환산 수익률과 변동성을 계산합니다."""
    ticker_list = [t.strip().upper() for t in ticker_list if t.strip()]
    if not ticker_list:
        return None, "티커를 입력해주세요."

    hist_data, error = load_historical_data(ticker_list, start_date, end_date)
    if error: return None, error
    
    if isinstance(hist_data.columns, pd.MultiIndex):
        returns = hist_data['Close'].pct_change().dropna(axis=0, how='all')
    else:
        # 단일 티커가 입력된 경우 (리스트지만 yf.download가 단일 DataFrame을 반환)
        returns = hist_data['Close'].pct_change().dropna()
        returns = pd.DataFrame(returns, columns=ticker_list)
        
    returns = returns.dropna(axis=1, how='all')

    if returns.empty or len(returns) < 20: 
        return None, "데이터 부족 또는 티커 오류로 수익률 계산 불가."
    
    annual_factor = 252
    mean_returns = returns.mean() * annual_factor
    annual_volatility = returns.std() * np.sqrt(annual_factor)
    
    metrics_list = []
    for ticker in returns.columns:
        metrics_list.append({
            'Ticker': ticker,
            'Return': mean_returns.get(ticker, 0.0),
            'Volatility': annual_volatility.get(ticker, 0.0)
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics['Sharpe_Ratio'] = df_metrics['Return'] / df_metrics['Volatility'].mask(df_metrics['Volatility'] == 0)
    # 수익률(Return) 기준으로 내림차순 정렬
    df_metrics = df_metrics.sort_values(by='Return', ascending=False).reset_index(drop=True)
    
    return df_metrics, None


@st.cache_data(ttl=3600)
def load_historical_data(ticker_or_list, start_date, end_date, max_retries=3, period=None):
    """yfinance에서 주가 데이터를 로드합니다. (단일/복수 티커 지원)"""
    if start_date == 'max':
        start_date = None
    
    if period == 'max':
        start_date = None

    for attempt in range(max_retries):
        try:
            # yf.download는 ticker_or_list가 리스트면 멀티 티커를 로드함
            hist = yf.download(ticker_or_list, start=start_date, end=end_date, period=period, progress=False)
            if hist.empty:
                return None, "해당 기간의 주가 데이터를 가져올 수 없습니다."
            return hist, None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
            else:
                return None, f"데이터 로드 중 오류가 발생했습니다: {e}"
    return None, "Unexpected failure in Historical Data loading."

# ==============================================================================
# 2. 핵심 계산 함수 (PER 및 보조 지표)
# (calculate_per_and_indicators와 run_dynamic_per_simulation는 유지)
# ==============================================================================

def calculate_per_and_indicators(df, eps):
    """PER, 이동평균선, 선형 추세선, PER 매력도 점수를 계산합니다."""
    # ... (기존 로직 유지) ...
    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        # 멀티 인덱스 DataFrame에서 'Close' 레벨과 첫 번째(유일한) 티커를 선택
        # df_calc는 사이드바의 단일 티커용이므로, 첫 번째 컬럼을 사용합니다.
        data['Price'] = data['Close'].iloc[:, 0]
    else:
        # 단일 인덱스 DataFrame인 경우 (기존 로직)
        data['Price'] = data['Close']
        

    data['EPS'] = eps
    data['PER'] = np.where(data['EPS'] > 0, data['Price'] / data['EPS'], np.inf)

    per_data_for_calc = data[data['PER'] != np.inf]

    ma_windows = [5, 20, 60, 120]
    for w in ma_windows:
        data[f'Price_MA_{w}'] = data['Price'].rolling(window=w).mean()

    if not per_data_for_calc.empty:
        # PER Trend
        x_values = np.arange(len(per_data_for_calc))
        slope_per, intercept_per, _, _, _ = linregress(x_values, per_data_for_calc['PER'])

        x_full = np.arange(len(data))
        data['PER_Trend'] = intercept_per + slope_per * x_full

        # PER Score
        valid_per_data = data.loc[per_data_for_calc.index].copy()
        data['PER_Residual'] = np.nan
        data.loc[valid_per_data.index, 'PER_Residual'] = valid_per_data['PER'] - valid_per_data['PER_Trend']

        per_sd = data['PER_Residual'].std()
        data['PER_SD'] = per_sd

        if per_sd > 0 and not data.empty:
            current_per = data['PER'].iloc[-1]
            current_trend = data['PER_Trend'].iloc[-1]
            z_score = (current_per - current_trend) / per_sd
            score = 100 * (1 - (z_score + 2) / 4)
            score = max(0, min(100, score))
            data['PER_Score'] = score
        else:
            data['PER_Score'] = np.nan
    else:
        data['PER_Trend'] = np.nan
        data['PER_Score'] = np.nan
        data['PER_Residual'] = np.nan
        data['PER_SD'] = np.nan

    # Price Trend
    x_values_price = np.arange(len(data))
    slope_price, intercept_price, _, _, _ = linregress(x_values_price, data['Price'])
    data['Price_Trend'] = intercept_price + slope_price * x_values_price

    return data

def run_dynamic_per_simulation(df_per_hist, initial_investment, initial_cash, regular_deposit, deposit_interval_days):
    """
    PER 기반 동적 매매 전략 시뮬레이션 (매매 대상: QQQ)
    (이 함수는 탭 1에서 사용하지 않지만, DCA 탭에서 QQQ 데이터를 로드하는 부분이 제거되어야 합니다.
    **참고:** 이 함수는 현재 앱에서 호출되지 않으므로 그대로 두지만, 실제로는 QQQ 데이터 로드 로직이 필요합니다.)
    """
    # ... (기존 로직 유지) ...
    # 전역 변수 참조: PER_CRITERIA_DYNAMIC, CASH_REINVESTMENT_RATIO, SELL_RATIO
    trading_dates = df_per_hist.index
    results = df_per_hist.copy()
    results['Shares'] = 0.0
    results['Cash_Pool'] = 0.0
    results['Total_Investment'] = 0.0
    results = results.dropna(subset=['QQQ_Price', 'Avg_PER'])

    if results.empty: return results

    # Initial Day
    initial_price = results['QQQ_Price'].iloc[0]
    results.loc[trading_dates[0], 'Shares'] = initial_investment / initial_price if initial_price > 0 else 0
    results.loc[trading_dates[0], 'Cash_Pool'] = initial_cash
    results.loc[trading_dates[0], 'Total_Investment'] = initial_investment
    last_deposit_date = trading_dates[0]

    for i in range(1, len(trading_dates)):
        current_date = trading_dates[i]
        prev_date = trading_dates[i - 1]
        prev_shares = results.loc[prev_date, 'Shares']
        prev_cash = results.loc[prev_date, 'Cash_Pool']
        prev_investment = results.loc[prev_date, 'Total_Investment']

        current_per = results.loc[current_date, 'Avg_PER']
        current_price = results.loc[current_date, 'QQQ_Price']

        deposit_added = 0
        is_trading_day = False
        if (current_date - last_deposit_date).days >= deposit_interval_days:
            deposit_added = regular_deposit
            last_deposit_date = current_date
            is_trading_day = True

        shares_change = 0
        cash_change = 0
        new_investment = prev_investment + deposit_added

        if is_trading_day:
            base_multiplier = 0
            reinvest_ratio = 0
            is_selling = False

            if current_per < PER_CRITERIA_DYNAMIC['BUY_3X']:
                base_multiplier = 3
                reinvest_ratio = CASH_REINVESTMENT_RATIO['BUY_3X']
            elif PER_CRITERIA_DYNAMIC['BUY_3X'] <= current_per < PER_CRITERIA_DYNAMIC['BUY_2X']:
                base_multiplier = 2
                reinvest_ratio = CASH_REINVESTMENT_RATIO['BUY_2X']
            elif PER_CRITERIA_DYNAMIC['BUY_2X'] <= current_per < PER_CRITERIA_DYNAMIC['BUY_1X']:
                base_multiplier = 1
                reinvest_ratio = CASH_REINVESTMENT_RATIO['BUY_1X']
            elif PER_CRITERIA_DYNAMIC['BUY_1X'] <= current_per < PER_CRITERIA_DYNAMIC['HOLD']:
                base_multiplier = 0
            elif current_per >= PER_CRITERIA_DYNAMIC['HOLD']:
                base_multiplier = 0
                is_selling = True

                sell_ratio = 0
                if PER_CRITERIA_DYNAMIC['HOLD'] <= current_per < PER_CRITERIA_DYNAMIC['SELL_15']:
                    sell_ratio = SELL_RATIO['SELL_15']
                elif PER_CRITERIA_DYNAMIC['SELL_15'] <= current_per < PER_CRITERIA_DYNAMIC['SELL_30']:
                    sell_ratio = SELL_RATIO['SELL_30']
                elif current_per >= PER_CRITERIA_DYNAMIC['SELL_30']:
                    sell_ratio = SELL_RATIO['SELL_50']

                if sell_ratio > 0 and prev_shares > 0:
                    shares_sold = prev_shares * sell_ratio
                    shares_change -= shares_sold
                    sell_value = shares_sold * current_price
                    cash_change += sell_value
                cash_change += deposit_added

            if base_multiplier > 0:
                pure_investment = deposit_added * base_multiplier
                reinvest_cash = prev_cash * reinvest_ratio
                total_buy_amount = pure_investment + reinvest_cash

                if current_price > 0:
                    shares_bought = total_buy_amount / current_price
                    shares_change += shares_bought

                cash_change -= reinvest_cash

            if base_multiplier == 0 and not is_selling:
                cash_change += deposit_added

        new_shares = prev_shares + shares_change
        new_cash = prev_cash + cash_change
        
        results.loc[current_date, 'Shares'] = new_shares
        results.loc[current_date, 'Cash_Pool'] = new_cash
        results.loc[current_date, 'Total_Investment'] = new_investment

    results['Stock_Value'] = results['Shares'] * results['QQQ_Price']
    results['Portfolio_Value'] = results['Stock_Value'] + results['Cash_Pool']
    results['Return'] = results['Portfolio_Value'] - results['Total_Investment']

    return results

def get_historical_per_series(tickers, start_date, end_date):
    """
    단일 또는 다중 티커에 대해 시가총액 가중 평균(또는 단일) PER 시계열을 반환합니다.
    섹션 1(빅테크 그룹)과 섹션 3(개별 종목)에서 공통으로 사용합니다.
    """
    try:
        # 이 함수 내부에서 'calculate_accurate_group_per_history'의 로직을 수행하거나 
        # 해당 함수를 호출하여 결과를 가져옵니다.
        # (이미 정의되어 있다고 가정하신 calculate_accurate_group_per_history 활용)
        series, error = calculate_accurate_group_per_history(
            tickers, start_date=start_date, end_date=end_date
        )
        return series, error
    except Exception as e:
        return None, str(e)

def calculate_stats(series):
    """PER 시계열에서 통계값(평균, 중앙값, 이상치 제거)을 계산합니다."""
    if series is None or series.empty:
        return None, None, None
    # 상위 2% 이상치 제거 후 통계 계산
    clean_series = series[series < series.quantile(0.98)]
    return clean_series, clean_series.mean(), clean_series.median()


def calculate_group_metrics(df, selected_tickers):
    """선택된 종목들의 시총, 순이익 합계 및 평균 PER을 계산"""
    selected_df = df[df['Ticker'].isin(selected_tickers)]
    total_market_cap = selected_df['MarketCap'].sum()
    total_net_income = selected_df['NetIncome'].sum()
    
    avg_per = total_market_cap / total_net_income if total_net_income != 0 else np.nan
    avg_per_str = f"{avg_per:,.2f}" if not np.isnan(avg_per) else "N/A"
    
    # get_per_color 함수가 정의되어 있다고 가정
    dynamic_color, position_text = get_per_color(avg_per) 
    
    return total_market_cap, total_net_income, avg_per, avg_per_str, position_text


# [전역 함수] 섹션 1과 섹션 3에서 공통으로 호출
def get_common_per_analysis(tickers, start, end):
    """
    그 시점의 실제 실적(Dynamic TTM)을 반영한 PER 시계열을 가져오고
    통계치(평균, 중앙값)를 계산하여 반환합니다.
    """
    # 1. 역사적 PER 시계열 데이터 가져오기 (기존에 정의한 함수 호출)
    series, error = calculate_accurate_group_per_history(tickers, start, end)
    
    if error or series is None or series.empty:
        return None, None, None, error

    # 2. 이상치 제거 (상위 2% 제거)
    clean_series = series[series < series.quantile(0.98)]
    avg_val = clean_series.mean()
    median_val = clean_series.median()
    
    return series, avg_val, median_val, None

# ==============================================================================
# 3. 유틸리티 및 포매팅 함수
# ==============================================================================

@st.cache_data
def format_value(val):
    """숫자를 T (조), B (십억) 단위로 포매팅합니다."""
    if pd.isna(val) or val == 0:
        return "-"
    if abs(val) >= 1e12:
        return f"{val / 1e12:,.2f}T"
    elif abs(val) >= 1e9:
        return f"{val / 1e9:,.2f}B"
    return f"{val:,.2f}"

def get_per_color(per_value):
    """PER 값에 따른 색상을 반환합니다."""
    # ... (기존 로직 유지) ...
    if np.isnan(per_value): return "gray", "N/A"
    
    if per_value < PER_CRITERIA_DYNAMIC['BUY_3X']:
        return "green", "3배 레버리지 매수 구간 (30 미만)"
    elif PER_CRITERIA_DYNAMIC['BUY_3X'] <= per_value < PER_CRITERIA_DYNAMIC['BUY_2X']:
        return "#90ee90", "2배 레버리지 매수 구간 (30 ~ 32)"
    elif PER_CRITERIA_DYNAMIC['BUY_2X'] <= per_value < PER_CRITERIA_DYNAMIC['BUY_1X']:
        return "blue", "1배 매수 구간 (32 ~ 35)"
    elif PER_CRITERIA_DYNAMIC['BUY_1X'] <= per_value < PER_CRITERIA_DYNAMIC['HOLD']:
        return "orange", "현금 보유 구간 (35 ~ 38)"
    elif PER_CRITERIA_DYNAMIC['HOLD'] <= per_value < PER_CRITERIA_DYNAMIC['SELL_15']:
        return "red", "3배 매도 구간 (38 ~ 41)"
    elif PER_CRITERIA_DYNAMIC['SELL_15'] <= per_value < PER_CRITERIA_DYNAMIC['SELL_30']:
        return "#8b0000", "2배 매도 구간 (41 ~ 45)"
    elif per_value >= PER_CRITERIA_DYNAMIC['SELL_30']:
        return "#8b0000", "매도 구간 (45 이상)"
    return "black", "N/A"

# Plotly PER 기준 가로선 추가 함수
def add_per_criteria_lines(fig, yaxis='y1'):
    """Plotly 그래프에 PER 기준선과 라벨을 추가합니다."""
    # ... (기존 로직 유지) ...
    for level in PER_LEVELS_SORTED:
        if level in PER_LINE_STYLES:
            color, label = PER_LINE_STYLES[level]

            fig.add_shape(
                type="line", xref="paper", yref=yaxis,
                x0=0, y0=level, x1=1, y1=level,
                line=dict(color=color, width=1, dash="dot"),
            )
            # 라벨은 가장 오른쪽 끝에 배치
            fig.add_annotation(
                x=1.00, y=level, yref=yaxis, xref="paper",
                text=label.split(' ')[0], showarrow=False,
                xanchor="right", yshift=5, font=dict(size=10, color=color),
            )
    return fig


# ==============================================================================
# 4. Streamlit UI 및 레이아웃 설정 (최상단)
# ==============================================================================

st.set_page_config(layout="wide", page_title="주식 분석 앱")

# --- 사이드바: 기본 설정 ---
with st.sidebar:
    st.header("⚙️ 기본 설정")
    # 1. 티커 입력
    ticker_symbol = st.text_input("주식 티커:", value="NVDA").upper()

    # 2. 기간 선택 로직
    period_options = {"1년": 365, "2년": 730, "5년": 1825, "YTD": 'ytd', "최대 기간": 'max'}
    selected_period_name = st.selectbox("기간 선택:", list(period_options.keys()), index=0)

    # --- [수정 포인트] 모든 조건에서 days와 start_date_default를 확실히 정의 ---
    if selected_period_name == 'ytd':
        start_date_default = date(TODAY.year, 1, 1)
        days = (TODAY - start_date_default).days
    elif selected_period_name == 'max':
        # 'max'의 경우 시스템상 아주 먼 과거(예: 20년 전)로 설정하거나 
        # yfinance가 인식하는 'max' 문자열을 위해 일수는 넉넉히 설정
        start_date_default = TODAY - timedelta(days=365*20) 
        days = 365*20
    else:
        # 1년, 2년, 5년 선택 시
        days = period_options.get(selected_period_name, 365)
        start_date_default = TODAY - timedelta(days=days)

    # 3. 날짜 입력 필드 (위에서 계산된 default값 사용)
    start_date_input = st.date_input("시작 날짜:", value=start_date_default, max_value=TODAY)
    end_date_input = st.date_input("최종 날짜:", value=TODAY, max_value=TODAY)

    # 최종 기간 결정 (문자열 형식으로 통일)
    # '최대 기간'을 선택했더라도 사용자가 날짜를 직접 수정했다면 그 날짜를 우선시함
    if selected_period_name == 'max' and start_date_input == start_date_default:
        start_date_final = 'max'
    else:
        start_date_final = start_date_input.strftime('%Y-%m-%d')
        
    end_date_final = end_date_input.strftime('%Y-%m-%d')

# ==============================================================================
# 5. 핵심 데이터 로드 및 전역 데이터프레임 준비
# ==============================================================================

# --- A. 개별 티커 정보 로드 (Tab 2, 3, 4용) ---
info, info_error = load_ticker_info(ticker_symbol)
if info_error:
    st.error(f"티커 정보를 가져오는 데 실패했습니다: {info_error}")
    st.stop()

# --- B. 주가 데이터 로드 (Tab 2, 3, 4용) ---
hist_data, data_error = load_historical_data(
    ticker_symbol,
    start_date=start_date_final,
    end_date=end_date_final
)
if data_error:
    st.error(f"데이터 로드 오류: {data_error}")
    st.stop()

# --- C. 핵심 계산 실행 (Tab 2, 3, 4에서 사용) ---
df_calc = calculate_per_and_indicators(hist_data, info['EPS'])





# --- D. 메뉴 설정 (PC/모바일 반응형 통합) ---
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "재무 분석" 

menu_options = [
    "재무 분석", "적립 모드 (DCA)", 
    "PER 그래프 분석", "주가 및 이동평균선", 
    "2 티커 최적", "다중 티커 비교"
]

# CSS: PC에서는 한 줄(6열), 모바일에서는 2열로 강제 고정
st.markdown("""
    <style>
    /* 1. 기본 설정 (PC 등 넓은 화면): 한 줄에 6개 배치 */
    div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 8px !important;
    }

    /* 2. 모바일 설정 (화면 너비 768px 이하): 강제 2열 그리드 */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            display: grid !important;
            grid-template-columns: 1fr 1fr !important; /* 무조건 2열 */
            gap: 6px !important;
        }
        
        div[data-testid="column"] {
            width: 100% !important;
            min-width: 0px !important;
            flex: none !important;
        }
        
        .stButton button p {
            font-size: 0.72rem !important;
        }
    }

    /* 공통 버튼 스타일 */
    .stButton button {
        height: 2.8rem !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 버튼 출력 (컴퓨터에서는 한 줄에 6개를 다 넣기 위해 단일 columns 생성)
# 모바일에서는 위 미디어 쿼리에 의해 알아서 그리드로 변합니다.
cols = st.columns(len(menu_options))
for i, option in enumerate(menu_options):
    with cols[i]:
        is_active = (st.session_state.active_tab == option)
        btn_type = "primary" if is_active else "secondary"
        if st.button(option, key=f"resp_btn_{i}", use_container_width=True, type=btn_type):
            st.session_state.active_tab = option
            st.rerun()

st.markdown("---")












































































































































# ==============================================================================
# 6. Tab 구현부
# ==============================================================================

# ------------------------------------------------------------------------------
# 섹션 1: 재무 분석 (빅테크)
# ------------------------------------------------------------------------------

if st.session_state.active_tab == "재무 분석":
    
    # 1. 초기 데이터 로드 및 세션 상태 관리
    tech_df_raw = load_big_tech_data(DEFAULT_BIG_TECH_TICKERS)
    
    if 'tech_select_state' not in st.session_state:
        # 모든 종목을 기본적으로 선택 상태로 초기화
        st.session_state['tech_select_state'] = {t: True for t in DEFAULT_BIG_TECH_TICKERS}

    # 2. [핵심] 현재 선택된 종목 기준 실시간 지표 계산 (NameError 방지)
    # 에디터의 변경사항을 반영하기 위해 현재 세션 상태의 티커들을 필터링합니다.
    selected_tickers = [t for t, selected in st.session_state['tech_select_state'].items() if selected]
    selected_df = tech_df_raw[tech_df_raw['Ticker'].isin(selected_tickers)]
    
    # 총합 및 평균 계산
    total_market_cap = selected_df['MarketCap'].sum()
    total_net_income = selected_df['NetIncome'].sum()
    
    if total_net_income > 0:
        average_per = total_market_cap / total_net_income
        average_per_str = f"{average_per:,.2f}"
        dynamic_color, position_text_raw = get_per_color(average_per)
    else:
        average_per = np.nan
        average_per_str = "N/A"
        dynamic_color, position_text_raw = "#gray", "데이터 없음"



    # 4. 역사적 PER 추이 그래프 (과거 시점 실적 반영 로직)
    # 전역적으로 정의된 시계열 계산 함수를 호출합니다.
    group_per_series, hist_error_tab1 = calculate_accurate_group_per_history(
        selected_tickers, start_date=start_date_final, end_date=end_date_final
    )
    
    if hist_error_tab1:
        st.warning(f"PER 추이 데이터를 로드할 수 없습니다: {hist_error_tab1}")
    elif group_per_series is None or group_per_series.empty:
        st.info("선택된 종목들의 유효한 데이터가 부족하여 그래프를 표시할 수 없습니다.")
    else:
        # 통계 계산 (이상치 제거)
        clean_per_values = group_per_series[group_per_series < group_per_series.quantile(0.98)]
        avg_per_hist = clean_per_values.mean()
        median_per_hist = clean_per_values.median()

        # Plotly 그래프 생성
        fig_per_tab1 = go.Figure()
        
        # 메인 가중 평균 PER 곡선
        fig_per_tab1.add_trace(go.Scatter(
            x=group_per_series.index, y=group_per_series, 
            mode='lines', name='시총 가중 평균 PER 추이',
            line=dict(color='#1f77b4', width=2),
            showlegend=False
        ))
        
        # 역사적 평균 및 중앙값 가로선
        fig_per_tab1.add_hline(y=avg_per_hist, line_dash="dash", line_color="#d62728", 
                               annotation_text=f"평균: {avg_per_hist:.2f}")
        fig_per_tab1.add_hline(y=median_per_hist, line_dash="dot", line_color="#ff7f0e", 
                               annotation_text=f"중앙값: {median_per_hist:.2f}")

        # 현재 시점 강조 점
        current_per_val = group_per_series.iloc[-1]
        fig_per_tab1.add_trace(go.Scatter(
            x=[group_per_series.index[-1]], y=[current_per_val],
            mode='markers', marker=dict(size=10, color='black'),
            name=f"현재: {current_per_val:.2f}"
        ))

# --- [수정 포인트] 모바일 최적화 및 범례 위치 변경 ---
        fig_per_tab1.update_layout(
            title="빅테크 그룹 가중 평균 PER 히스토리",
            xaxis_title="날짜", 
            yaxis_title="PER",
            hovermode="x unified", 
            template="plotly_white", 
            height=500,
            # 범례 설정: 상단 내부로 이동
            legend=dict(
                orientation="h",       # 가로 방향 배치
                yanchor="bottom",      # y축 기준점을 아래로
                y=1.02,                # 그래프 바로 위(안쪽 상단은 0.9 정도로 조절 가능)
                xanchor="right",       # x축 기준점을 오른쪽으로
                x=1,                   # 오른쪽 끝에 밀착
                bgcolor="rgba(255, 255, 255, 0.5)" # 배경 반투명 처리
            ),
            margin=dict(l=10, r=10, t=50, b=10) # 모바일 여백 최소화
        )
        st.plotly_chart(fig_per_tab1, use_container_width=True)
        
    st.markdown("---")
    # 3. 상단 요약 Metric 표시
    # 그래프보다 위에 배치하여 사용자가 변경사항을 즉시 숫자로 확인하게 합니다.
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric(
            label="선택 종목 평균 PER (TTM)", 
            value=average_per_str, 
            delta=position_text_raw if average_per_str != "N/A" else None, 
            delta_color='off'
        )
    with col_sum2:
        st.metric(label="총 시가총액 합", value=format_value(total_market_cap))
    with col_sum3:
        st.metric(label="총 순이익 합 (역산)", value=format_value(total_net_income))

    st.markdown("---")
    
    # 5. 하단: 투자 기준 표와 종목 편집기 (1:2 비율)
    col_criteria, col_editor = st.columns([1, 2])
    
    with col_criteria:
        # 투자 기준 정의
        investment_criteria = pd.DataFrame({
            "PER 범위": ["< 30", "30 ~ 32", "32 ~ 35", "35 ~ 38", "38 ~ 41", "41 ~ 45", ">= 45"],
            "권장 조치": ["3배 레버리지 매수", "2배 레버리지 매수", "1배 매수", "현금 보유", "3배 매도", "2배 매도", "매도"]
        })

        def highlight_criteria(s):
            """현재 평균 PER 위치에 하이라이트 적용"""
            if np.isnan(average_per): return [''] * len(s)
            per_range = s['PER 범위'].replace(' ', '')
            is_match = False
            try:
                if '<' in per_range:
                    if average_per < float(per_range.split('<')[1]): is_match = True
                elif '~' in per_range:
                    low, high = map(float, per_range.split('~'))
                    if low <= average_per < high: is_match = True
                elif '>=' in per_range:
                    if average_per >= float(per_range.split('>=')[1]): is_match = True
            except: pass
            
            return [f'background-color: {dynamic_color}; color: white; font-weight: bold;'] * len(s) if is_match else [''] * len(s)

        st.markdown(f"**현재 평균 PER : {average_per_str}**")
        st.dataframe(
            investment_criteria.style.apply(highlight_criteria, axis=1),
            hide_index=True, height=280, use_container_width=True
        )

    with col_editor:
        # 편집용 데이터프레임 구성
        editor_df = tech_df_raw.copy()
        editor_df['Select'] = editor_df['Ticker'].apply(lambda t: st.session_state['tech_select_state'].get(t, True))
        editor_df['PER (TTM)'] = editor_df['TrailingPE'].apply(lambda x: f"{x:.2f}" if x > 0 else "-")
        editor_df['시가총액 (USD)'] = editor_df['MarketCap'].apply(format_value)
        editor_df['순이익 (USD)'] = editor_df['NetIncome'].apply(format_value)

        st.markdown("**분석 포함 종목 선택**", help="체크를 해제하면 전체 평균 계산에서 제외됩니다.")
        
        edited_df = st.data_editor(
            editor_df[['Select', 'Ticker', '시가총액 (USD)', 'PER (TTM)', '순이익 (USD)']],
            column_config={
                "Select": st.column_config.CheckboxColumn("선택"),
                "Ticker": st.column_config.TextColumn(disabled=True),
                "시가총액 (USD)": st.column_config.TextColumn(disabled=True),
                "PER (TTM)": st.column_config.TextColumn(disabled=True),
                "순이익 (USD)": st.column_config.TextColumn(disabled=True),
            },
            hide_index=True,
            key='big_tech_editor_v2'
        )
        
        # 에디터 변경사항을 세션 상태에 저장하여 다음 렌더링 시 반영
        new_selections = {row['Ticker']: row['Select'] for _, row in edited_df.iterrows()}
        if new_selections != st.session_state['tech_select_state']:
            st.session_state['tech_select_state'] = new_selections
            st.rerun() # 변경 즉시 상단 메트릭과 그래프를 갱신하기 위해 rerun 호출

# ------------------------------------------------------------------------------
# 섹션 2: 적립 모드 (DCA 시뮬레이션)
# ------------------------------------------------------------------------------
elif st.session_state.active_tab == "적립 모드 (DCA)":

    # --- 1. 시뮬레이션 설정 ---
    if 'dca_amount' not in st.session_state: st.session_state.dca_amount = 10.0
    if 'dca_freq' not in st.session_state: st.session_state.dca_freq = "매일"

    deposit_amount = st.session_state.dca_amount
    deposit_frequency = st.session_state.dca_freq

    # --- 2. 시뮬레이션 계산 (DCA 로직) ---
    dca_df = df_calc.copy()
    dca_df['WeekOfYear'] = dca_df.index.isocalendar().week.astype(int)
    dca_df['Month'] = dca_df.index.month

    if deposit_frequency == "매일": invest_dates = dca_df.index
    elif deposit_frequency == "매주": invest_dates = dca_df.groupby('WeekOfYear')['Price'].head(1).index
    elif deposit_frequency == "매월": invest_dates = dca_df.groupby('Month')['Price'].head(1).index

    dca_result = dca_df[dca_df.index.isin(invest_dates)].copy()
    dca_result['Shares_Bought'] = deposit_amount / dca_result['Price']
    dca_result['Total_Shares'] = dca_result['Shares_Bought'].cumsum()
    dca_result['Cumulative_Investment'] = np.arange(1, len(dca_result) + 1) * deposit_amount

    # 전체 기간에 걸쳐 결과 전파
    full_dca_results = dca_df.copy()
    full_dca_results['Total_Shares'] = dca_result['Total_Shares'].reindex(dca_df.index, method='ffill').fillna(0)
    full_dca_results['Cumulative_Investment'] = dca_result['Cumulative_Investment'].reindex(dca_df.index, method='ffill').fillna(0)
    full_dca_results['Current_Value'] = full_dca_results['Total_Shares'] * full_dca_results['Price']

    # --- 3. 그래프 생성 ---
    fig_dca = go.Figure()

    fig_dca.add_trace(go.Scatter(x=full_dca_results.index, y=full_dca_results['Price'], mode='lines', name='주가 추이 (배경)',
                                 line=dict(color='gray', width=1), opacity=0.3, yaxis='y2'))

    fig_dca.add_trace(go.Scatter(x=full_dca_results.index, y=full_dca_results['Current_Value'], mode='lines', name='현재 평가 가치',
                                 line=dict(color='green', width=2), yaxis='y1'))

    fig_dca.add_trace(go.Scatter(x=full_dca_results.index, y=full_dca_results['Cumulative_Investment'], mode='lines', name='총 투자 금액',
                                 line=dict(color='red', width=2, dash='dash'), yaxis='y1'))

    fig_dca.update_layout(
        title=f"{ticker_symbol} 적립식 투자(DCA) 시뮬레이션", height=500, xaxis_title="날짜", hovermode="x unified",
        legend=dict(x=0.01, y=0.99, yanchor="top", xanchor="left"),
        yaxis=dict(title=dict(text="투자 금액/가치 (USD)", font=dict(color="green")), side="left", showgrid=True),
        yaxis2=dict(title=dict(text="주가 (Price, 배경)", font=dict(color="gray")), overlaying="y", side="right", showgrid=False, range=[full_dca_results['Price'].min() * 0.9, full_dca_results['Price'].max() * 1.1])
    )
    st.plotly_chart(fig_dca, use_container_width=True)
    

    # --- 4. 시뮬레이션 설정 (그래프 아래) ---
    st.markdown("---")
    st.markdown("### 🛠️ 시뮬레이션 설정")
    col_dca_config1, col_dca_config2 = st.columns(2)
    with col_dca_config1:
        st.number_input("**적립 금액 (USD)**", min_value=1.0, step=1.0, format="%.2f", key='dca_amount', help="매번 투자할 금액을 입력합니다.")
    with col_dca_config2:
        current_freq_index = ["매일", "매주", "매월"].index(st.session_state.dca_freq)
        st.selectbox("**적립 주기**", ["매일", "매주", "매월"], index=current_freq_index, key='dca_freq')

    # --- 5. 최종 요약 (가장 아래) ---
    st.markdown("---")
    st.markdown("### 📊 최종 요약")

    if not full_dca_results.empty:
        final_row = full_dca_results.iloc[-1]
        current_value = final_row['Current_Value'].item()
        cumulative_investment = final_row['Cumulative_Investment'].item()
        col_dca_summary = st.columns(4)
        col_dca_summary[0].metric(label="최종 평가 가치", value=f"${current_value:,.2f}", delta=f"${current_value - cumulative_investment:,.2f}")
        col_dca_summary[1].metric("총 투자 금액", f"${cumulative_investment:,.2f}")
        col_dca_summary[2].metric("총 매수 주식 수", f"{final_row['Total_Shares'].item():,.4f} 주")


# ------------------------------------------------------------------------------
# 섹션 3: PER 그래프 분석
# ------------------------------------------------------------------------------

elif st.session_state.active_tab == "PER 그래프 분석":
    
    # --- 전역 함수 호출 (단일 티커 전달) ---
    with st.spinner(f"{ticker_symbol}의 역사적 PER 데이터를 분석 중..."):
        single_per_series, hist_error_tab3 = get_historical_per_series([ticker_symbol], start_date_final, end_date_final)
    
    if hist_error_tab3:
        st.warning("역사적 PER 데이터를 불러올 수 없습니다. (ETF 등 실적 데이터가 없는 티커일 수 있습니다.)")
    elif single_per_series is not None and not single_per_series.empty:
        
        # 통계 계산
        clean_series, avg_per, median_per = calculate_stats(single_per_series)
        current_per = single_per_series.iloc[-1]

        # 그래프 생성
        fig_per = go.Figure()
        fig_per.add_trace(go.Scatter(
            x=single_per_series.index, y=single_per_series, 
            mode='lines', name='역사적 PER (TTM)',
            line=dict(color='#1f77b4'),
            showlegend=False
        ))
        
        # 평균선/중앙값선 추가
        fig_per.add_hline(y=avg_per, line_dash="dash", line_color="red", annotation_text=f"평균: {avg_per:.2f}")
        fig_per.add_hline(y=median_per, line_dash="dot", line_color="orange", annotation_text=f"중앙값: {median_per:.2f}")

        fig_per.update_layout(title=f"{ticker_symbol} 역사적 PER 추이 (Dynamic TTM)", template="plotly_white")
        st.plotly_chart(fig_per, use_container_width=True)

        # 매력도 점수 (현재 PER이 역사적 평균 대비 어디에 있는지 계산)
        st.markdown("### 📊 현재 PER 매력도")
        std_per = clean_series.std()
        z_score = (current_per - avg_per) / std_per if std_per != 0 else 0
        score = max(0, min(100, 100 * (1 - (z_score + 2) / 4))) # 간단한 상대 점수화
        
        st.metric(label="현재 PER 매력도 점수", value=f"{score:.0f} 점", 
                  delta=f"현재 PER: {current_per:.2f}", delta_color="off")
        st.info("💡 이 점수는 과거 5년(또는 설정 기간) 평균 PER 대비 현재 위치를 나타냅니다.")
# ------------------------------------------------------------------------------
# 섹션 4: 주가 그래프 및 이동평균선/추세선
# ------------------------------------------------------------------------------
elif st.session_state.active_tab == "주가 및 이동평균선":

    # --- 1. Session State 초기화 및 값 로드 ---
    if 'price_overlay_key_visible' not in st.session_state: st.session_state.price_overlay_key_visible = "이평선 (이동평균선)"
    if 'price_ma_window_key_visible' not in st.session_state: st.session_state.price_ma_window_key_visible = 20

    price_overlay_choice = st.session_state.price_overlay_key_visible
    price_ma_window = st.session_state.price_ma_window_key_visible

    if price_overlay_choice == "이평선 (이동평균선)":
        overlay_column_price = f'Price_MA_{price_ma_window}'
        overlay_name_price = f'{price_ma_window}일 이동평균'
        if overlay_column_price not in df_calc.columns:
            # 5, 20, 60, 120 외의 기간은 여기서 계산
            df_calc[overlay_column_price] = df_calc['Price'].rolling(window=price_ma_window).mean()
    else:
        overlay_column_price = 'Price_Trend'
        overlay_name_price = '주가 선형 추세선'

    # --- 2. 주가 그래프 생성 ---

    fig_price = go.Figure()

    # 종가 (Price)
    fig_price.add_trace(go.Scatter(x=df_calc.index, y=df_calc['Price'], mode='lines', name='종가 (Price)', line=dict(color='blue', width=1.5)))

    # 보조선 (MA 또는 추세선)
    fig_price.add_trace(go.Scatter(x=df_calc.index, y=df_calc[overlay_column_price], mode='lines', name=overlay_name_price, line=dict(color='red', dash='dash', width=2)))

    fig_price.update_layout(
        title=f"{ticker_symbol} 주가 추이", height=500, xaxis_title="날짜", yaxis_title="주가 (Price)",
        hovermode="x unified", template="plotly_white", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        showlegend=False
    )
    st.plotly_chart(fig_price, use_container_width=True)
    

    # --- 3. 위젯 재배치 (그래프 아래 - 화면 표시용) ---
    st.markdown("---")
    st.markdown("### 🛠️ 보조선 설정 (위 그래프에 적용)")

    col_config_bottom1, col_config_bottom2 = st.columns(2)

    with col_config_bottom1:
        st.selectbox("**보조선 선택**", ["선형 추세선", "이평선 (이동평균선)"], key='price_overlay_key_visible')

    if st.session_state.price_overlay_key_visible == "이평선 (이동평균선)":
        with col_config_bottom2:
            st.number_input("**이평선 기간 (일)**", min_value=1, max_value=300, step=5, key='price_ma_window_key_visible', format="%d", help="차트에 표시할 이동평균선의 기간을 설정합니다.")
    else:
        with col_config_bottom2:
            st.markdown(" ")
######################################################################################






elif st.session_state.active_tab == "2 티커 최적":


    
    # ------------------------------------
    # 1. 입력 섹션 (생략)
    # ------------------------------------
    col_input_tickers, col_input_period = st.columns([2, 1])

    with col_input_tickers:
        ticker_input_str = st.text_input("비교할 티커 입력 (쉼표나 공백으로 구분)", value="SCHD QQQ", key="tickers_mpt_single_sec5")
        
    with col_input_period:
        period_options_mpt = {"1년": 365, "3년": 3 * 365, "5년": 5 * 365}
        selected_period_name = st.selectbox("분석 기간:", list(period_options_mpt.keys()), index=1, key="period_mpt_sec5")
        
    ticker_list = [t.strip().upper() for t in ticker_input_str.split() if t.strip()]
    if len(ticker_list) >= 2:
        ticker1_mpt, ticker2_mpt = ticker_list[0], ticker_list[1]
    else:
        ticker1_mpt, ticker2_mpt = "", ""

    days_mpt = period_options_mpt[selected_period_name]
    start_date_mpt = (TODAY - timedelta(days=days_mpt)).strftime('%Y-%m-%d')
    end_date_mpt = TODAY.strftime('%Y-%m-%d')


    # ------------------------------------
    # 2. 분석 로직 (가중치 관련 로직 제거)
    # ------------------------------------
    if ticker1_mpt and ticker2_mpt and ticker1_mpt != ticker2_mpt:
        
        with st.spinner(f"**{ticker1_mpt}**와 **{ticker2_mpt}**의 포트폴리오 분석 중..."):
            # calculate_portfolio_metrics 함수는 내부적으로 여전히 가중치를 계산하지만,
            # 여기서는 반환되는 df_port와 key_points에서 가중치 정보를 사용하지 않습니다.
            df_port, port_error, key_points = calculate_portfolio_metrics(ticker1_mpt, ticker2_mpt, start_date_mpt, end_date_mpt)
            
        if port_error:
            st.error(f"포트폴리오 데이터 로드 오류: {port_error}")
        elif df_port is not None and not df_port.empty:
            
            mvp = key_points['mvp']
            max_sharpe = key_points['max_sharpe']
            asset_metrics = key_points['asset_metrics']
            
            # 개별 자산의 100% 포트폴리오 지점 데이터
            asset1_100_pt = df_port.loc[df_port['Weight_1'].idxmax()]
            asset2_100_pt = df_port.loc[df_port['Weight_2'].idxmax()]
            
            # --- Plotly 그래프 생성 (Efficient Frontier) ---

            
            fig_mpt = go.Figure()
            
            # 1. 시뮬레이션된 포트폴리오 (라인)
            fig_mpt.add_trace(go.Scatter(
                x=df_port['Volatility'] * 100, y=df_port['Return'] * 100,
                mode='lines', marker=dict(size=4, color='lightgray'),
                name='포트폴리오 배합', line=dict(color='gray', width=1),
                # 가중치 정보 제거: 수익률, 위험, 샤프 비율만 표시
                customdata=df_port[['Return', 'Volatility', 'Sharpe_Ratio']].values * np.array([100, 100, 1]),
                hovertemplate=('수익률: %{customdata[0]:.2f}%<br>위험: %{customdata[1]:.2f}%<br>' +
                               'Sharpe Ratio: %{customdata[2]:.2f}<extra></extra>'),
                showlegend=False,
            ))
            
            # 2. 개별 자산
            fig_mpt.add_trace(go.Scatter(
                x=[asset_metrics[ticker1_mpt]['Volatility'] * 100, asset_metrics[ticker2_mpt]['Volatility'] * 100],
                y=[asset_metrics[ticker1_mpt]['Return'] * 100, asset_metrics[ticker2_mpt]['Return'] * 100],
                mode='markers+text', name='개별 자산',
                marker=dict(size=12, color='darkorange'),
                text=[ticker1_mpt, ticker2_mpt], textposition="bottom right",
                
                # 가중치 정보 제거: 티커 이름, 수익률, 위험만 표시
                customdata=np.array([[asset_metrics[ticker1_mpt]['Return'] * 100, asset_metrics[ticker1_mpt]['Volatility'] * 100],
                                     [asset_metrics[ticker2_mpt]['Return'] * 100, asset_metrics[ticker2_mpt]['Volatility'] * 100]]),
                hovertemplate=('자산: %{text}<br>수익률: %{customdata[0]:.2f}%<br>위험: %{customdata[1]:.2f}%<extra></extra>'),
                showlegend=False
            ))
            
            # 3. 주요 지점 강조 (MVP, Max Sharpe)
            key_points_data = [(mvp, '최소 분산 (MVP)', 'blue'), (max_sharpe, '최대 샤프 비율', 'green')]
            for point, name, color in key_points_data:
                
                point_return, point_volatility = point['Return'] * 100, point['Volatility'] * 100
                point_sharpe = point['Sharpe_Ratio']
                    
                fig_mpt.add_trace(go.Scatter(
                    x=[point_volatility], y=[point_return], mode='markers', name=name,
                    marker=dict(size=15, color=color, symbol='star'),
                    hovertemplate=(
                        f'<b>{name}</b><br>수익률: {point_return:.2f}%<br>위험: {point_volatility:.2f}%<br>' +
                        f'Sharpe Ratio: {point_sharpe:.2f}<extra></extra>') # 가중치 제거
                ))

            fig_mpt.update_layout(
                title=f"포트폴리오 효율적 투자선 ({ticker1_mpt} vs. {ticker2_mpt})", 
                xaxis_title="연간 변동성 (위험, %)", yaxis_title="연간 수익률 (%)",
                template="plotly_white", height=500, hovermode="closest",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                showlegend=False
            )
            st.plotly_chart(fig_mpt, use_container_width=True)
            
            # --- 결과 요약 ---
            st.markdown("### 🎯 주요 포트폴리오 분석 결과")
            
            # 1. 개별 자산 메트릭
            st.markdown("#### 개별 자산 분석")
            col_asset_1_r, col_asset_1_v, col_asset_2_r, col_asset_2_v = st.columns(4)
            
            col_asset_1_r.metric(f"📈 {ticker1_mpt} 수익률", f"{asset_metrics[ticker1_mpt]['Return'] * 100:.2f}%")
            col_asset_1_v.metric(f"위험", f"{asset_metrics[ticker1_mpt]['Volatility'] * 100:.2f}%")
            
            col_asset_2_r.metric(f"📈 {ticker2_mpt} 수익률", f"{asset_metrics[ticker2_mpt]['Return'] * 100:.2f}%")
            col_asset_2_v.metric(f"위험", f"{asset_metrics[ticker2_mpt]['Volatility'] * 100:.2f}%")
            
            st.markdown("---")
            
            # 2. MVP와 Max Sharpe 출력 (가중치 제거)
            col_mvp, col_sharpe = st.columns(2)
            
            with col_mvp:
                st.subheader("🛡️ 최소 분산 (MVP)")
                st.metric(f"수익률", f"{mvp['Return'] * 100:.2f}%")
                st.metric(f"변동성 (위험)", f"{mvp['Volatility'] * 100:.2f}%")
                st.metric(f"Sharpe Ratio", f"{mvp['Sharpe_Ratio']:.2f}", help="무위험 이자율 0 가정 시")

            with col_sharpe:
                st.subheader("🌟 최대 샤프 비율")
                st.metric(f"Sharpe Ratio", f"{max_sharpe['Sharpe_Ratio']:.2f}")
                st.metric(f"수익률", f"{max_sharpe['Return'] * 100:.2f}%")
                st.metric(f"변동성 (위험)", f"{max_sharpe['Volatility'] * 100:.2f}%")
                
            st.markdown("---")
            
            # 3. 개념 설명 및 다이어그램
            st.subheader("💡 효율적 투자선 (Efficient Frontier)의 개념")
            st.markdown(
                """
                **효율적 투자선**은 **주어진 위험(변동성)에서 최대의 기대 수익률을 제공**하거나, **주어진 기대 수익률에서 최소의 위험을 제공**하는 포트폴리오들의 집합입니다. 
                * **최소 분산 포트폴리오 (MVP):** 포트폴리오가 달성할 수 있는 가장 낮은 위험(변동성)을 가진 지점입니다.
                * **최대 샤프 비율 포트폴리오:** 위험 한 단위당 가장 높은 초과 수익(샤프 비율)을 제공하는 지점입니다. 이는 자본 시장선(CML)과 효율적 투자선이 접하는 지점입니다.
                """
            )

            st.info(f"⚠️ **참고:** 분석 기간: {start_date_mpt} ~ {end_date_mpt}. 모든 수익률 및 변동성은 연환산 기준이며, 무위험 이자율은 0으로 가정합니다.")

        else:
            st.warning("유효한 데이터를 찾지 못했거나 오류가 발생했습니다. 티커와 기간을 확인해 주세요.")
    else:
        st.info("공백으로 구분된 서로 다른 두 개의 유효한 주식 티커를 입력하고 분석 기간을 선택해 주세요.")




# --------------------------------------------------------------------------
# 섹션 6: 다중 티커 단순 비교 (그래프 상단 배치 및 Zoom Out 기능 재확인)
# --------------------------------------------------------------------------
elif st.session_state.active_tab == "다중 티커 비교":

    col_multi_input, col_multi_period = st.columns([2, 1])

    with col_multi_input:
        multi_ticker_input = st.text_input("비교할 티커 입력 (쉼표나 공백으로 구분)", value="TQQQ QQQ SPY", key="multi_ticker_mpt_sec6")
        
    with col_multi_period:
        period_options_multi = {"1년": 365, "3년": 3 * 365, "5년": 5 * 365}
        selected_period_multi_name = st.selectbox("분석 기간:", list(period_options_multi.keys()), index=0, key="period_mpt_sec6")

    ticker_list_multi = [t.strip().upper() for t in multi_ticker_input.replace(',', ' ').split() if t.strip()]

    days_multi = period_options_multi[selected_period_multi_name]
    start_date_multi = (TODAY - timedelta(days=days_multi)).strftime('%Y-%m-%d')
    end_date_multi = TODAY.strftime('%Y-%m-%d')

    if ticker_list_multi:
        with st.spinner(f"다중 티커 ({', '.join(ticker_list_multi)}) 분석 중..."):
            df_multi_metrics, multi_error = calculate_multi_ticker_metrics(ticker_list_multi, start_date_multi, end_date_multi)
            
        if multi_error:
            st.error(f"다중 티커 분석 오류: {multi_error}")
        elif df_multi_metrics is not None and not df_multi_metrics.empty:
            

            
            # ==========================================================
            # 2. Plotly 그래프 (수익률 vs 위험률 Scatter) - 맨 위로 이동
            # ==========================================================

            
            fig_multi = go.Figure()

            fig_multi.add_trace(go.Scatter(
                x=df_multi_metrics['Volatility'] * 100,
                y=df_multi_metrics['Return'] * 100,
                mode='markers+text',
                text=df_multi_metrics['Ticker'],
                textposition="bottom center",
                marker=dict(
                    size=15, 
                    opacity=0.8, 
                    color=df_multi_metrics['Sharpe_Ratio'], 
                    colorscale='Viridis', 
                    showscale=True, 
                    # ⭐ 컬러바를 하단으로 이동시키는 핵심 설정 ⭐
                    colorbar=dict(
                        title="Sharpe Ratio",
                        orientation="h",      # 가로 방향(Horizontal)
                        yanchor="top",        # 기준점을 위쪽으로
                        y=-0.2,               # 그래프 x축 아래로 배치
                        thickness=15,         # 막대 두께 조절
                        len=0.7               # 막대 길이 (70%)
                    )
                ),
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    '수익률: %{y:.2f}%<br>' +
                    '위험률: %{x:.2f}%<br>' +
                    '샤프 비율: %{marker.color:.2f}<extra></extra>'
                )
            ))

            fig_multi.update_layout(

                xaxis_title="연간 위험률 (%)", 
                yaxis_title="연간 수익률 (%)",
                template="plotly_white", 
                height=500, 
                hovermode="closest",
                # ⭐ Zoom Out/기본 뷰 강화를 위한 설정 ⭐
                xaxis=dict(autorange=True, rangemode='tozero'), # 0부터 시작하도록 설정
                yaxis=dict(autorange=True, rangemode='tozero') # 0부터 시작하도록 설정
            )
            st.plotly_chart(fig_multi, use_container_width=True)
            
            
            # ==========================================================
            # 3. 결과표 출력 - 그래프 아래에 배치
            # ==========================================================

            # DataFrame 포매팅 및 순위 지정
            df_display = df_multi_metrics.copy()
            df_display = df_display.sort_values(by='Sharpe_Ratio', ascending=False)
            df_display.index = range(1, len(df_display) + 1)
            df_display.index.name = "순위"
            
            # 표시 형식 지정
            df_display['Return'] = df_display['Return'].apply(lambda x: f"{x * 100:.2f}%")
            df_display['Volatility'] = df_display['Volatility'].apply(lambda x: f"{x * 100:.2f}%")
            df_display['Sharpe_Ratio'] = df_display['Sharpe_Ratio'].apply(lambda x: f"{x:.2f}")

            st.dataframe(
                df_display.rename(columns={'Return': '연간 수익률', 'Volatility': '연간 위험률', 'Sharpe_Ratio': '샤프 비율'}),
                use_container_width=True,
            )

            st.info(f"⚠️ **참고:** 분석 기간: {start_date_multi} ~ {end_date_multi}. 샤프 비율 계산 시 무위험 이자율은 편의상 0으로 가정했습니다.")
            
        else:
            st.info("유효한 데이터를 가진 티커가 없습니다. 티커를 확인해 주세요.")
    else:
        st.info("비교할 티커들을 입력해 주세요.")








