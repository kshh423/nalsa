import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from datetime import date, timedelta
import time # 재시도 대기 시간 확보를 위해 time 모듈 import

# --- 0. 상수 정의 ---
DEFAULT_BIG_TECH_TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA']

# PER 기준 상수 (Tab 5 동적 매매 전략용)
PER_CRITERIA_DYNAMIC = {
    'BUY_3X': 30.0,
    'BUY_2X': 32.0,
    'BUY_1X': 35.0,
    'HOLD': 38.0,
    'SELL_15': 41.0,
    'SELL_30': 45.0,
    'SELL_50': 45.0
}

# 현금 재투자 비율
CASH_REINVESTMENT_RATIO = {
    'BUY_3X': 0.50,
    'BUY_2X': 0.30,
    'BUY_1X': 0.10
}

# 매도 비율
SELL_RATIO = {
    'SELL_15': 0.15,
    'SELL_30': 0.30,
    'SELL_50': 0.50
}


# --- 1. 데이터 로드 및 캐싱 함수 (TTL=3600 적용 및 재시도 로직 강화) ---

@st.cache_data(ttl=3600) # 👈 1시간 캐싱 적용
def load_ticker_info(ticker, max_retries=3):
    """티커 정보를 로드합니다 (EPS, 회사 이름) - 재시도 로직 포함."""
    
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker)
            info = data.info

            # EPS (Trailing EPS 선호, 없으면 Forward EPS 시도)
            eps = info.get('trailingEps')
            if eps is None or eps == 0:
                eps = info.get('forwardEps')

            per_info = {
                'EPS': eps if eps else 0,
                'CompanyName': info.get('longName', ticker),
            }
            # 성공적으로 데이터를 가져오면 즉시 반환
            return per_info, None
        
        except Exception as e:
            # 마지막 시도가 아니면 재시도
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1) # 1차: 5초, 2차: 10초 대기
                print(f"[{ticker}] Ticker info load failed (Attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # 모든 시도 실패 시 오류 반환
                return None, f"Ticker information could not be loaded after {max_retries} attempts: {e}"

    return None, "Unexpected failure in Ticker Info loading." # 안전 장치


@st.cache_data(ttl=3600) # 👈 1시간 캐싱 적용
def load_historical_data(ticker, start_date, end_date, max_retries=3):
    """yfinance에서 주가 데이터를 로드합니다 (재시도 로직 포함)."""
    if start_date == 'max':
        start_date = None

    for attempt in range(max_retries):
        try:
            hist = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if hist.empty:
                # 데이터는 가져왔지만 내용이 비어있는 경우
                return None, "해당 기간의 주가 데이터를 가져올 수 없습니다."
            return hist, None
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"[{ticker}] Historical data load failed (Attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None, f"데이터 로드 중 오류가 발생했습니다: {e}"
    
    return None, "Unexpected failure in Historical Data loading."


@st.cache_data(ttl=3600) # 👈 1시간 캐싱 적용
def load_big_tech_data(tickers, max_retries=3):
    """요청된 빅테크 종목의 재무 정보를 로드합니다 (재시도 로직 포함)."""
    data_list = []
    
    for ticker in tickers:
        for attempt in range(max_retries):
            try:
                info = yf.Ticker(ticker).info
                market_cap = info.get('marketCap', np.nan)
                trailing_pe = info.get('trailingPE', np.nan)

                # Net Income = Market Cap / PER
                net_income = market_cap / trailing_pe if market_cap and trailing_pe and trailing_pe > 0 else np.nan

                data_list.append({
                    'Ticker': ticker,
                    'MarketCap': market_cap,
                    'TrailingPE': trailing_pe,
                    'NetIncome': net_income,
                })
                break # 성공하면 다음 티커로 이동
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 3 * (attempt + 1) # 개별 티커는 3초 간격으로 재시도
                    print(f"[{ticker}] Big Tech info load failed (Attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # 모든 시도 실패 시 NaN 값으로 처리하고 다음 티커로 이동
                    print(f"[{ticker}] Failed to load info after {max_retries} attempts.")
                    data_list.append({
                        'Ticker': ticker,
                        'MarketCap': np.nan,
                        'TrailingPE': np.nan,
                        'NetIncome': np.nan,
                    })
                    break

    return pd.DataFrame(data_list)


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


# --- 2. PER 및 보조 지표 계산 함수 (기존 함수 유지) ---

def calculate_per_and_indicators(df, eps):
    """PER, 이동평균선, 선형 추세선, PER 매력도 점수를 계산합니다."""
    data = df.copy()
    data['Price'] = data['Close']

    # 1. PER 계산
    data['EPS'] = eps
    data['PER'] = np.where(data['EPS'] > 0, data['Price'] / data['EPS'], np.inf)

    per_data_for_calc = data[data['PER'] != np.inf]

    # 2. 이동평균선 계산 (주가만 계산)
    ma_windows = [5, 20, 60, 120]
    for w in ma_windows:
        data[f'Price_MA_{w}'] = data['Price'].rolling(window=w).mean()

    # 3. 선형 추세선 계산 (PER과 Price 모두 계산)
    if not per_data_for_calc.empty:
        # PER Trend
        x_values = np.arange(len(per_data_for_calc))
        slope_per, intercept_per, _, _, _ = linregress(x_values, per_data_for_calc['PER'])

        x_full = np.arange(len(data))
        data['PER_Trend'] = intercept_per + slope_per * x_full

        # 4. PER 매력도 점수 계산 (선형 추세선 괴리 기반)

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


# --- 3. 동적 매매 시뮬레이션 로직 (Tab 5 전용) ---

@st.cache_data(ttl=3600) # 👈 1시간 캐싱 적용
def load_historical_per_and_qqq_data(tickers, start_date, end_date, max_retries=3):
    """
    선택된 빅테크 종목들의 가중 평균 PER 시계열과 QQQ 가격을 계산하여 반환합니다.
    (Tab 5 시뮬레이션용)
    """
    target_tickers = list(set(tickers + ['QQQ']))
    
    # 주가 데이터 로드 재시도
    price_data_all = None
    hist_error = None
    
    for attempt in range(max_retries):
        try:
            price_data_all = yf.download(target_tickers, start=start_date, end=end_date, progress=False)['Close']
            if price_data_all.empty:
                hist_error = "주가 데이터를 가져올 수 없습니다."
            if isinstance(price_data_all, pd.Series):
                price_data_all = price_data_all.to_frame(name=target_tickers[0])
            
            if not hist_error:
                break # 성공
                
        except Exception as e:
            hist_error = f"주가 데이터 로드 오류: {e}"
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"[Multi Tickers] Historical data load failed (Attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None, hist_error
    
    if hist_error:
         return None, hist_error

    qqq_price_series = price_data_all['QQQ']

    # 1. 고정 EPS 정보 및 시가총액 정보 로드 (가중 평균 PER을 위한 EPS와 MarketCap)
    eps_data = {}
    market_caps = {}
    valid_tickers = []

    # ⚠️ yfinance에서 실시간 Market Cap을 가져와 시가총액 가중 평균 PER의 근사치 계산에 사용
    # 참고: 이 정보 로딩은 load_big_tech_data와 유사하게 TTL=3600으로 캐시되지만,
    # 해당 함수에서 개별 티커 정보 로드 시 Rate Limit 오류가 발생할 수 있습니다.
    
    # 이 부분의 반복적인 yf.Ticker().info 호출은 위 load_big_tech_data 함수 로직에서 이미
    # 재시도 로직을 포함하므로, 여기서는 그대로 사용하고 캐싱 TTL에 의존합니다.
    
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            eps = info.get('trailingEps')
            if eps is None or eps == 0:
                eps = info.get('forwardEps')
            market_cap = info.get('marketCap', 0)

            if eps and eps > 0 and market_cap > 0:
                eps_data[ticker] = eps
                market_caps[ticker] = market_cap
                valid_tickers.append(ticker)
        except:
            continue

    if not valid_tickers:
        return None, "선택된 종목들에서 유효한 EPS나 Market Cap을 찾을 수 없어 PER 계산이 불가능합니다."

    # 2. 가중 평균 PER 시계열 계산 (MarketCap 대신 Price Sum을 EPS Sum으로 나누는 방식 채택)
    total_eps_fixed = sum(eps_data.values())

    price_sum_data = price_data_all[valid_tickers].sum(axis=1, skipna=True)
    approx_per_series = price_sum_data / total_eps_fixed

    df_result = pd.DataFrame({
        'Avg_PER': approx_per_series,
        'QQQ_Price': qqq_price_series
    }).dropna(subset=['Avg_PER', 'QQQ_Price'])

    return df_result, None


def run_dynamic_per_simulation(df_per_hist, initial_investment, initial_cash, regular_deposit, deposit_interval_days):
    """
    PER 기반 동적 매매 전략 시뮬레이션 (매매 대상: QQQ)
    """
    trading_dates = df_per_hist.index

    results = df_per_hist.copy()

    results['Shares'] = 0.0 # 보유 QQQ 주식 수
    results['Cash_Pool'] = 0.0 # 현금 풀
    results['Total_Investment'] = 0.0 # 총 누적 기본 적립금

    results = results.dropna(subset=['QQQ_Price', 'Avg_PER'])

    if results.empty:
        return results

    # 첫 날 초기화 (첫날은 매매 실행일로 간주)
    initial_price = results['QQQ_Price'].iloc[0]
    results.loc[trading_dates[0], 'Shares'] = initial_investment / initial_price if initial_price > 0 else 0
    results.loc[trading_dates[0], 'Cash_Pool'] = initial_cash
    results.loc[trading_dates[0], 'Total_Investment'] = initial_investment

    last_deposit_date = trading_dates[0]

    for i in range(1, len(trading_dates)):
        current_date = trading_dates[i]
        prev_date = trading_dates[i - 1]

        # 이전 날짜의 상태를 다음 날로 계승
        prev_shares = results.loc[prev_date, 'Shares']
        prev_cash = results.loc[prev_date, 'Cash_Pool']
        prev_investment = results.loc[prev_date, 'Total_Investment']

        current_per = results.loc[current_date, 'Avg_PER']
        current_price = results.loc[current_date, 'QQQ_Price']

        # 1. 정기 적립금 체크 (매매 실행 여부 결정)
        deposit_added = 0
        is_trading_day = False
        if (current_date - last_deposit_date).days >= deposit_interval_days:
            deposit_added = regular_deposit
            last_deposit_date = current_date
            is_trading_day = True # 적립 주기가 도래한 날에만 매매 실행

        shares_change = 0
        cash_change = 0
        new_investment = prev_investment + deposit_added

        # --------------------------------------------------------
        # 2. 매매 로직 실행 (매매 주기가 도래한 날에만!)
        # --------------------------------------------------------
        if is_trading_day:

            base_multiplier = 0 # 매수 멀티플라이어 (0: HOLD/SELL)
            reinvest_ratio = 0
            is_selling = False

            # --- 매수/재투자 구간 (PER < 35) ---
            if current_per < PER_CRITERIA_DYNAMIC['BUY_3X']:
                base_multiplier = 3
                reinvest_ratio = CASH_REINVESTMENT_RATIO['BUY_3X']
            elif PER_CRITERIA_DYNAMIC['BUY_3X'] <= current_per < PER_CRITERIA_DYNAMIC['BUY_2X']:
                base_multiplier = 2
                reinvest_ratio = CASH_REINVESTMENT_RATIO['BUY_2X']
            elif PER_CRITERIA_DYNAMIC['BUY_2X'] <= current_per < PER_CRITERIA_DYNAMIC['BUY_1X']:
                base_multiplier = 1
                reinvest_ratio = CASH_REINVESTMENT_RATIO['BUY_1X']

            # --- 현금 보유 구간 (35 <= PER < 38) ---
            elif PER_CRITERIA_DYNAMIC['BUY_1X'] <= current_per < PER_CRITERIA_DYNAMIC['HOLD']:
                base_multiplier = 0 # HOLD

            # --- 매도 구간 (PER >= 38) ---
            elif current_per >= PER_CRITERIA_DYNAMIC['HOLD']:
                base_multiplier = 0 # SELL
                is_selling = True

                sell_ratio = 0
                if PER_CRITERIA_DYNAMIC['HOLD'] <= current_per < PER_CRITERIA_DYNAMIC['SELL_15']:
                    sell_ratio = SELL_RATIO['SELL_15']
                elif PER_CRITERIA_DYNAMIC['SELL_15'] <= current_per < PER_CRITERIA_DYNAMIC['SELL_30']:
                    sell_ratio = SELL_RATIO['SELL_30']
                elif current_per >= PER_CRITERIA_DYNAMIC['SELL_30']:
                    sell_ratio = SELL_RATIO['SELL_50']

                    # 복리적 매도 로직
                if sell_ratio > 0 and prev_shares > 0:
                    shares_sold = prev_shares * sell_ratio
                    shares_change -= shares_sold
                    sell_value = shares_sold * current_price
                    cash_change += sell_value
                # 매도/현금 보유 구간에서는 정기 적립금은 Cash Pool에 적립
                cash_change += deposit_added

            # 3. 매수/재투자 실행 (base_multiplier > 0일 때)
            if base_multiplier > 0:
                pure_investment = deposit_added * base_multiplier
                reinvest_cash = prev_cash * reinvest_ratio
                total_buy_amount = pure_investment + reinvest_cash

                if current_price > 0:
                    shares_bought = total_buy_amount / current_price
                    shares_change += shares_bought

                cash_change -= reinvest_cash
                # 매수 구간에서는 정기 적립금이 Shares로 변환되었음.

            # 4. 현금 보유 구간 (base_multiplier == 0, 매도 아닐 때)
            if base_multiplier == 0 and not is_selling:
                cash_change += deposit_added

        # --------------------------------------------------------
        # 5. 결과 업데이트 (매매를 했든 안 했든 주식 수와 현금은 갱신)
        # --------------------------------------------------------
        new_shares = prev_shares + shares_change
        new_cash = prev_cash + cash_change
        # new_investment는 이미 위에서 계산됨 (매매 주기가 아니어도 deposit_added는 0)

        results.loc[current_date, 'Shares'] = new_shares
        results.loc[current_date, 'Cash_Pool'] = new_cash
        results.loc[
            current_date, 'Total_Investment'] = new_investment # 매매 주기가 아니면 deposit_added=0이므로 prev_investment 유지

    # 최종 가치 계산
    results['Stock_Value'] = results['Shares'] * results['QQQ_Price']
    results['Portfolio_Value'] = results['Stock_Value'] + results['Cash_Pool']
    results['Return'] = results['Portfolio_Value'] - results['Total_Investment']

    return results

# --- 4. Streamlit UI 및 레이아웃 설정 ---

st.set_page_config(layout="wide", page_title="주식 분석 앱")

# --- 기간 설정 (기본값) ---
TODAY = date.today()
ONE_YEAR_AGO = TODAY - timedelta(days=365)

# --- 사이드바: 기본 설정 ---
with st.sidebar:
    st.header("⚙️ 기본 설정")

    # 3-1. 티커 입력 (기본값 NVDA)
    ticker_symbol = st.text_input(
        "**주식 티커를 입력하세요:**",
        value="NVDA",
        help="이 탭에 표시되는 티커는 탭 2, 3, 4의 분석 대상이 됩니다."
    ).upper()

    # 3-2. 기간 선택 드롭다운
    period_options = {
        "1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "2년": 730, "5년": 1825, "YTD (연초 대비)": 'ytd', "최대 기간": 'max'
    }
    selected_period_name = st.selectbox(
        "**기간 선택 (드롭다운):**",
        list(period_options.keys()),
        index=3
    )

    # 3-3. 시작 날짜 기본값 설정
    if selected_period_name == 'ytd':
        start_date_default = date(TODAY.year, 1, 1)
    elif selected_period_name == 'max':
        start_date_default = ONE_YEAR_AGO
    else:
        days = period_options[selected_period_name]
        start_date_default = TODAY - timedelta(days=days)

    start_date_input = st.date_input(
        "**시작 날짜 (직접 입력):**",
        value=start_date_default,
        min_value=date(1900, 1, 1),
        max_value=TODAY
    )

    end_date_input = st.date_input(
        "**최종 날짜:**",
        value=TODAY,
        min_value=date(1900, 1, 1),
        max_value=TODAY
    )

    # 최종 기간 결정 로직
    if selected_period_name == 'max':
        start_date_final = 'max'
    elif selected_period_name == 'ytd':
        start_date_final = date(TODAY.year, 1, 1).strftime('%Y-%m-%d')
    else:
        start_date_final = start_date_input.strftime('%Y-%m-%d')

    end_date_final = end_date_input.strftime('%Y-%m-%d')

# --- 데이터 로드 (분석 대상 티커) ---
# load_ticker_info에 재시도 로직 포함
info, info_error = load_ticker_info(ticker_symbol)

if info_error:
    st.error(f"티커 정보를 가져오는 데 실패했습니다: {info_error}")
    st.stop()

st.subheader(f"🚀 {info['CompanyName']} ({ticker_symbol}) 분석")

# load_historical_data에 재시도 로직 포함
hist_data, data_error = load_historical_data(
    ticker_symbol,
    start_date=start_date_final,
    end_date=end_date_final
)

if data_error:
    st.error(f"데이터 로드 오류: {data_error}")
    st.stop()

# 최종 데이터 계산 (Tab 3, 4용)
df_calc = calculate_per_and_indicators(hist_data, info['EPS'])

# --- 5. 2x2 네모 박스 메뉴 구현 (Tab 5 추가) ---

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "재무 분석" # 초기 선택 메뉴

menu_options = [
    "재무 분석",
    "적립 모드 (DCA)",
    "PER 그래프 분석",
    "주가 및 이동평균선",
    "PER 기반 QQQ 동적 매매 시뮬레이터" # <<< Tab 5 추가
]
# 5개의 메뉴 버튼을 3x2 레이아웃으로 변경 (첫 줄 3개, 둘째 줄 2개)
cols_row1 = st.columns(3)
cols_row2 = st.columns(2)
cols = cols_row1 + cols_row2 + [None] # 총 5개 버튼 컬럼 + 남은 공간

for i, option in enumerate(menu_options):
    with cols[i]:
        if option == "PER 기반 QQQ 동적 매매 시뮬레이터":
            button_label = "PER 기반 QQQ 동적 매매 시뮬레이터"
        else:
            button_label = option

        is_active = (st.session_state.active_tab == option)
        button_type = "primary" if is_active else "secondary"

        if st.button(
                button_label,
                key=f"tab_button_{i}",
                use_container_width=True,
                type=button_type
        ):
            st.session_state.active_tab = option
            st.rerun()

st.markdown("---")
# ==============================================================================
# 섹션 1: 재무 분석 (빅테크)
# ==============================================================================
if st.session_state.active_tab == "재무 분석":

    BIG_TECH_TICKERS = DEFAULT_BIG_TECH_TICKERS

    # 데이터 로드 (캐싱된 함수 사용)
    tech_df_raw = load_big_tech_data(BIG_TECH_TICKERS)

    # 1. 체크박스(선택) 컬럼을 추가하여 Data Editor에 사용

    # Session State 초기화 (모두 True로 설정)
    if 'tech_select_state' not in st.session_state:
        initial_state = {t: True for t in BIG_TECH_TICKERS}
        st.session_state['tech_select_state'] = initial_state

    # 세션 상태에서 현재 선택 상태를 가져와 DataFrame에 반영
    editor_df = tech_df_raw.copy()
    editor_df['Select'] = editor_df['Ticker'].apply(lambda t: st.session_state['tech_select_state'].get(t, True))

    # PER (TTM)과 같은 원본 재무 데이터 포매팅
    editor_df['PER (TTM)'] = editor_df['TrailingPE'].apply(lambda x: f"{x:.2f}" if x > 0 else "-")
    editor_df['시가총액 (USD)'] = editor_df['MarketCap'].apply(format_value)
    editor_df['순이익 (USD, 역산)'] = editor_df['NetIncome'].apply(format_value)

    # 2. 체크된 종목만 필터링하여 합계 및 평균 계산
    selected_tickers = editor_df[editor_df['Select'] == True]['Ticker'].tolist()

    # 원본 데이터(MarketCap, TrailingPE, NetIncome)를 필터링
    selected_df = tech_df_raw[tech_df_raw['Ticker'].isin(selected_tickers)]

    total_market_cap = selected_df['MarketCap'].sum()
    total_net_income = selected_df['NetIncome'].sum()

    # 시가총액 가중 평균 PER 계산
    average_per = total_market_cap / total_net_income if total_net_income != 0 else np.nan
    average_per_str = f"{average_per:.2f}" if not np.isnan(average_per) else "N/A"  # PER 문자열 포매팅

    # --- 1. 투자 기준 표 (Highlighting 포함) 생성 (최상단) ---

    # 현재 평균 PER에 맞는 동적 색상 결정 로직
    dynamic_color = "black"
    if not np.isnan(average_per):
        if average_per < 30:
            dynamic_color = "green"
        elif 30 <= average_per < 32:
            dynamic_color = "#90ee90"  # 연두색
        elif 32 <= average_per < 35:
            dynamic_color = "blue"
        elif 35 <= average_per < 38:
            dynamic_color = "orange"
        elif 38 <= average_per < 41:
            dynamic_color = "red"
        elif 41 <= average_per < 45:
            dynamic_color = "#8b0000"  # 어두운 빨간색
        elif average_per >= 45:
            dynamic_color = "#8b0000"

    # 헤더에 동적 색상 적용
    st.markdown(
        f"### 🎯 평균 PER 기반 투자 기준 (평균 per : <span style='color:{dynamic_color};'>{average_per_str}</span>)",
        unsafe_allow_html=True
    )

    investment_criteria = pd.DataFrame({
        "PER 범위": ["< 30", "30 ~ 32", "32 ~ 35", "35 ~ 38", "38 ~ 41", "41 ~ 45", ">= 45"],
        "권장 조치": ["3배 레버리지 매수", "2배 레버리지 매수", "1배 매수", "현금 보유", "3배 매도", "2배 매도", "매도"]
    })


    # 하이라이트 스타일 정의 함수
    def highlight_criteria(s):
        if np.isnan(average_per):
            return [''] * len(s)

        is_highlight = False
        per_range = s['PER 범위'].replace(' ', '')

        try:
            if '<' in per_range:
                upper = float(per_range.split('<')[1])
                if average_per < upper:
                    is_highlight = True
            elif '~' in per_range:
                lower, upper = map(float, per_range.split('~'))
                if lower <= average_per < upper:
                    is_highlight = True
            elif '>=' in per_range:
                lower = float(per_range.split('>=')[1])
                if average_per >= lower:
                    is_highlight = True
        except:
            is_highlight = False  # 에러 방지

        color_code = "black"  # 기본값
        if not np.isnan(average_per):
            if average_per < 30:
                color_code = "green"
            elif 30 <= average_per < 32:
                color_code = "#90ee90"
            elif 32 <= average_per < 35:
                color_code = "blue"
            elif 35 <= average_per < 38:
                color_code = "orange"
            elif 38 <= average_per < 41:
                color_code = "red"
            elif 41 <= average_per < 45:
                color_code = "#8b0000"
            elif average_per >= 45:
                color_code = "#8b0000"

        if is_highlight:
            # 하이라이트 배경 색상은 위에서 결정된 color_code 사용
            return [f'background-color: {color_code}; color: white; font-weight: bold;'] * len(s)
        else:
            return [''] * len(s)


    # 기존의 PER 값 포함 마크다운은 유지
    st.markdown(f"""
        <p style='font-size: small; color: gray;'>
        🤔 최하단 표 체크 시 평균 반영 (현재 선택 종목 평균 PER : **{average_per_str}**)
        </p>
    """, unsafe_allow_html=True)

    st.dataframe(
        investment_criteria.style.apply(highlight_criteria, axis=1),
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 📉 선택 종목 합계 및 평균 지표")

    # --- 합계 및 평균 Metric 표시 ---
    col_sum1, col_sum2, col_sum3 = st.columns(3)

    col_sum1.metric(
        label="총 시가총액 합",
        value=format_value(total_market_cap)
    )
    col_sum2.metric(
        label="총 순이익 합 (역산)",
        value=format_value(total_net_income)
    )

    # --- 평균 PER 위치 안내 로직 (Metric 아래 델타 색상 결정 로직) ---
    position_text_raw = ""
    color_code = "black"  # 이 color_code는 metric 아래에 별도로 표시되는 텍스트의 색상을 결정합니다.

    if not np.isnan(average_per):
        if average_per < 30:
            position_text_raw = "3배 레버리지 매수 구간 (30 미만)";
            color_code = "green"
        elif 30 <= average_per < 32:
            position_text_raw = "2배 레버리지 매수 구간 (30 ~ 32)";
            color_code = "#90ee90"
        elif 32 <= average_per < 35:
            position_text_raw = "1배 매수 구간 (32 ~ 35)";
            color_code = "blue"
        elif 35 <= average_per < 38:
            position_text_raw = "현금 보유 구간 (35 ~ 38)";
            color_code = "orange"
        elif 38 <= average_per < 41:
            position_text_raw = "3배 매도 구간 (38 ~ 41)";
            color_code = "red"
        elif 41 <= average_per < 45:
            position_text_raw = "2배 매도 구간 (41 ~ 45)";
            color_code = "#8b0000"
        elif average_per >= 45:
            position_text_raw = "매도 구간 (45 이상)";
            color_code = "#8b0000"

    # st.metric 호출
    col_sum3.metric(
        label="선택 종목 평균 PER (TTM)",
        value=average_per_str,
        delta=position_text_raw if position_text_raw else None,
        delta_color='off'
    )

    # metric 아래에 위치 안내를 HTML로 재표시하여 색상 적용
    if position_text_raw:
        delta_html = f"<span style='color: {color_code}; font-weight: bold;'>{position_text_raw}</span>"
        st.markdown(delta_html, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📋 개별 종목 데이터 편집")

    # st.data_editor를 사용하여 체크박스를 포함한 표 출력
    edited_df = st.data_editor(
        editor_df[['Select', 'Ticker', '시가총액 (USD)', 'PER (TTM)', '순이익 (USD, 역산)']],
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "선택",
                help="평균 PER 및 총합 계산에 포함할 종목을 선택하세요.",
            ),
            "Ticker": st.column_config.TextColumn(disabled=True),
            "시가총액 (USD)": st.column_config.TextColumn(disabled=True),
            "PER (TTM)": st.column_config.TextColumn(disabled=True),
            "순이익 (USD, 역산)": st.column_config.TextColumn(disabled=True),
        },
        hide_index=True,
        key='big_tech_editor'
    )

    # 체크박스 변경 시 상태를 즉시 Session State에 반영
    current_selections = {row['Ticker']: row['Select'] for index, row in edited_df.iterrows()}
    st.session_state['tech_select_state'] = current_selections

    # ++++++++++++++++ [추가된 부분] ++++++++++++++++
    st.markdown("---")
    st.markdown("### 📊 선택 종목 평균 PER 추이 및 매매 기준")

    # 1. 시계열 데이터 로드
    # Tab 5에서 사용하는 load_historical_per_and_qqq_data를 재사용하여 Avg_PER만 가져옴.
    # 단, 재무 분석 탭이므로, 사이드바에서 선택된 Ticker (NVDA)는 무시하고,
    # 'selected_tickers' (체크박스에서 선택된 종목)만 사용합니다.

    # 사이드바의 기간 설정을 그대로 사용
    avg_per_hist_tab1, hist_error_tab1 = load_historical_per_and_qqq_data(
        selected_tickers,
        start_date=start_date_final,
        end_date=end_date_final
    )

    if hist_error_tab1:
        st.warning(f"PER 추이 데이터를 로드할 수 없습니다: {hist_error_tab1}")
    elif avg_per_hist_tab1.empty or avg_per_hist_tab1['Avg_PER'].isnull().all():
        st.info("선택된 종목들의 유효한 PER 시계열 데이터가 부족하여 그래프를 표시할 수 없습니다.")
    else:
        # 2. 그래프 생성
        fig_per_tab1 = go.Figure()

        per_series = avg_per_hist_tab1['Avg_PER'].dropna()

        # 가중 평균 PER 추이
        fig_per_tab1.add_trace(go.Scatter(
            x=per_series.index,
            y=per_series,
            mode='lines',
            name='가중 평균 PER 추이',
            line=dict(color='blue', width=2),
            yaxis='y1'
        ))

        # PER 기준 가로선 추가 (Tab 5의 기준 재사용)
        per_line_styles = {
            PER_CRITERIA_DYNAMIC['BUY_3X']: ('green', '30.0 (3X 매수)'),
            PER_CRITERIA_DYNAMIC['BUY_2X']: ('darkgreen', '32.0 (2X 매수)'),
            PER_CRITERIA_DYNAMIC['BUY_1X']: ('blue', '35.0 (1X 매수)'),
            PER_CRITERIA_DYNAMIC['HOLD']: ('orange', '38.0 (HOLD)'),
            PER_CRITERIA_DYNAMIC['SELL_15']: ('red', '41.0 (15% 매도)'),
            PER_CRITERIA_DYNAMIC['SELL_30']: ('darkred', '45.0 (30~50% 매도)')
        }

        per_levels_sorted = sorted(list(set(PER_CRITERIA_DYNAMIC.values())))

        for level in per_levels_sorted:
            if level in [PER_CRITERIA_DYNAMIC['BUY_3X'], PER_CRITERIA_DYNAMIC['BUY_2X'], PER_CRITERIA_DYNAMIC['BUY_1X'],
                         PER_CRITERIA_DYNAMIC['HOLD'], PER_CRITERIA_DYNAMIC['SELL_15'],
                         PER_CRITERIA_DYNAMIC['SELL_30']]:
                color, label = per_line_styles.get(level, ('gray', f'{level:.1f}'))

                fig_per_tab1.add_shape(
                    type="line", xref="paper", yref="y1",
                    x0=0, y0=level, x1=1, y1=level,
                    line=dict(color=color, width=1, dash="dot"),
                )
                fig_per_tab1.add_annotation(
                    x=per_series.index[-1], y=level, yref="y1",
                    text=label.split(' ')[0], showarrow=False,
                    xanchor="left", yshift=5, font=dict(size=10, color=color)
                )

        # 레이아웃 설정
        fig_per_tab1.update_layout(
            title="선택 종목 가중 평균 PER 추이 및 매매 기준선 ",
            height=450,
            xaxis_title="날짜",
            yaxis_title="가중 평균 PER",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(x=0.01, y=0.99, yanchor="top", xanchor="left")
        )
        st.plotly_chart(fig_per_tab1, use_container_width=True)

        st.markdown(f"**현재 PER:** <span style='color:{dynamic_color}; font-weight: bold;'>{average_per_str}</span>",
                    unsafe_allow_html=True)
# ==============================================================================
# 섹션 2: 적립 모드 (DCA 시뮬레이션) - (기존 코드 유지)
# ==============================================================================
elif st.session_state.active_tab == "적립 모드 (DCA)":

    # --- 1. Session State 초기화 (위젯 값이 없을 때만 실행) ---
    if 'dca_amount' not in st.session_state:
        st.session_state.dca_amount = 10.0
    if 'dca_freq' not in st.session_state:
        st.session_state.dca_freq = "매일"

    # --- 2. 시뮬레이션 계산 (그래프를 그리기 위한 사전 계산) ---
    deposit_amount = st.session_state.dca_amount
    deposit_frequency = st.session_state.dca_freq

    dca_df = df_calc.copy()
    dca_df['DayOfYear'] = dca_df.index.dayofyear
    dca_df['WeekOfYear'] = dca_df.index.isocalendar().week.astype(int)
    dca_df['Month'] = dca_df.index.month

    if deposit_frequency == "매일":
        invest_dates = dca_df.index
    elif deposit_frequency == "매주":
        invest_dates = dca_df.groupby('WeekOfYear')['Price'].head(1).index
    elif deposit_frequency == "매월":
        invest_dates = dca_df.groupby('Month')['Price'].head(1).index

    dca_result = dca_df[dca_df.index.isin(invest_dates)].copy()

    dca_result['Shares_Bought'] = deposit_amount / dca_result['Price']
    dca_result['Total_Shares'] = dca_result['Shares_Bought'].cumsum()

    dca_result['Cumulative_Investment'] = np.arange(1, len(dca_result) + 1) * deposit_amount
    dca_result['Current_Value'] = dca_result['Total_Shares'] * dca_df['Price'].loc[dca_result.index]

    full_dca_results = dca_df.copy()
    full_dca_results['Total_Shares'] = dca_result['Total_Shares'].reindex(dca_df.index, method='ffill').fillna(0)
    full_dca_results['Cumulative_Investment'] = dca_result['Cumulative_Investment'].reindex(dca_df.index,
                                                                                            method='ffill').fillna(0)
    full_dca_results['Current_Value'] = full_dca_results['Total_Shares'] * full_dca_results['Price']

    # --- 3. 그래프 생성 ---
    fig_dca = go.Figure()

    fig_dca.add_trace(go.Scatter(
        x=full_dca_results.index, y=full_dca_results['Price'],
        mode='lines', name='주가 추이 (배경)',
        line=dict(color='gray', width=1), opacity=0.3, yaxis='y2'
    ))

    fig_dca.add_trace(go.Scatter(
        x=full_dca_results.index, y=full_dca_results['Current_Value'],
        mode='lines', name='현재 평가 가치',
        line=dict(color='green', width=2), yaxis='y1'
    ))

    fig_dca.add_trace(go.Scatter(
        x=full_dca_results.index, y=full_dca_results['Cumulative_Investment'],
        mode='lines', name='총 투자 금액',
        line=dict(color='red', width=2, dash='dash'), yaxis='y1'
    ))

    fig_dca.update_layout(
        title=f"{ticker_symbol} 적립식 투자(DCA) 시뮬레이션",
        height=500,
        xaxis_title="날짜",
        legend=dict(x=0.01, y=0.99, yanchor="top", xanchor="left"),
        hovermode="x unified",

        yaxis=dict(
            title=dict(
                text="투자 금액/가치 (USD)",
                font=dict(color="green")
            ),
            side="left",
            showgrid=True,
            zeroline=False,
            tickfont=dict(color="green"),
        ),
        yaxis2=dict(
            title=dict(
                text="주가 (Price, 배경)",
                font=dict(color="gray")
            ),
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="gray"),
            range=[full_dca_results['Price'].min() * 0.9, full_dca_results['Price'].max() * 1.1]
        )
    )
    st.plotly_chart(fig_dca, use_container_width=True)

    # --- 4. 시뮬레이션 설정 (그래프 아래) ---
    st.markdown("---")
    st.markdown("### 🛠️ 시뮬레이션 설정")

    col_dca_config1, col_dca_config2 = st.columns(2)

    with col_dca_config1:
        st.number_input(
            "**적립 금액 (USD)**",
            min_value=1.0,
            step=1.0,
            format="%.2f",
            key='dca_amount',
            help="매번 투자할 금액을 입력합니다."
        )

    with col_dca_config2:
        current_freq_index = ["매일", "매주", "매월"].index(st.session_state.dca_freq)

        st.selectbox(
            "**적립 주기**",
            ["매일", "매주", "매월"],
            index=current_freq_index,
            key='dca_freq',
        )

    # --- 5. 최종 요약 (가장 아래) ---
    st.markdown("---")
    st.markdown("### 📊 최종 요약")

    if not full_dca_results.empty:
        final_row = full_dca_results.iloc[-1]

        current_value = final_row['Current_Value'].item()
        cumulative_investment = final_row['Cumulative_Investment'].item()

        col_dca_summary = st.columns(4)
        col_dca_summary[0].metric(
            label="최종 평가 가치",
            value=f"${current_value:,.2f}",
            delta=f"${current_value - cumulative_investment:,.2f}"
        )
        col_dca_summary[1].metric("총 투자 금액", f"${cumulative_investment:,.2f}")
        col_dca_summary[2].metric("총 매수 주식 수", f"{final_row['Total_Shares'].item():,.4f} 주")


# ==============================================================================
# 섹션 3: PER 그래프 분석 - (기존 코드 유지)
# ==============================================================================
elif st.session_state.active_tab == "PER 그래프 분석":

    per_data_filtered = df_calc[df_calc['PER'] != np.inf]

    if per_data_filtered.empty:
        st.warning("PER 계산을 위한 유효한 EPS 데이터가 없거나, EPS가 0 이하입니다. ETF가 아닌 실제 기업의 Ticker를 입력해주세요.")
    else:
        # --- 그래프 생성 (PER 및 선형 추세선) (최상단) ---

        overlay_column = 'PER_Trend'
        overlay_name = 'PER 선형 추세선'

        hover_data = per_data_filtered.copy()

        # 각 시점의 Z-Score와 매력도 점수를 재계산 (NaN 처리 포함)
        hover_data['Calculated_Z_Score'] = (hover_data['PER'] - hover_data['PER_Trend']) / hover_data['PER_SD']
        hover_data['Calculated_Score'] = 100 * (1 - (hover_data['Calculated_Z_Score'] + 2) / 4)

        hover_data['Display_Score'] = hover_data['Calculated_Score'].apply(lambda s: max(0, min(100, s))).round(
            0)
        hover_data['Display_PER'] = hover_data['PER'].round(2)

        fig_per = go.Figure()

        # 1. 일별 PER
        fig_per.add_trace(go.Scatter(
            x=hover_data.index, y=hover_data['PER'],
            mode='lines', name='일별 PER',
            line=dict(color='blue', width=1.5),
            hovertemplate=(
                    '<b>날짜:</b> %{x|%Y-%m-%d}<br>' +
                    '<b>PER:</b> %{customdata[0]:.2f}<br>' +
                    '<b>매력도 점수:</b> %{customdata[1]:.0f}점 <extra></extra>'
            ),
            customdata=np.stack((hover_data['Display_PER'], hover_data['Display_Score']), axis=-1)
        ))

        # 2. PER 선형 추세선 (커서 정보는 표시하지 않음)
        fig_per.add_trace(go.Scatter(
            x=hover_data.index, y=hover_data[overlay_column],
            mode='lines', name=overlay_name,
            line=dict(color='red', dash='dash', width=2),
            hoverinfo='none'
        ))

        fig_per.update_layout(
            title=f"{ticker_symbol} 일별 PER 추이 (EPS: {info['EPS']:.2f} 기준)",
            height=500,
            xaxis_title="날짜",
            yaxis_title="PER (주가수익비율)",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_per, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 현재 PER 매력도")

        # --- 매력도 점수 표시 ---
        current_score = df_calc['PER_Score'].iloc[-1]

        if not np.isnan(current_score):
            st.metric(
                label="현재 PER 매력도 점수 (100점에 가까울수록 저평가)",
                value=f"{current_score:.0f} 점"
            )
        else:
            st.warning("PER 매력도 점수를 계산하기에 데이터가 부족하거나 EPS가 0 이하입니다.")

        st.info(f"⚠️ PER은 고정된 EPS ({info['EPS']:.2f})를 기반으로 계산되었으며, 주가 변동에 따른 PER 추이를 나타냅니다. (매수 추천 기준: 75점 이상)")


# ==============================================================================
# 섹션 4: 주가 그래프 및 이동평균선/추세선 - (기존 코드 유지)
# ==============================================================================
elif st.session_state.active_tab == "주가 및 이동평균선":

    # --- 1. Session State 초기화 및 값 로드 (그래프 계산에 사용) ---
    if 'price_overlay_key_visible' not in st.session_state:
        st.session_state.price_overlay_key_visible = "이평선 (이동평균선)"
    if 'price_ma_window_key_visible' not in st.session_state:
        st.session_state.price_ma_window_key_visible = 20

    price_overlay_choice = st.session_state.price_overlay_key_visible
    price_ma_window = st.session_state.price_ma_window_key_visible

    if price_overlay_choice == "이평선 (이동평균선)":
        overlay_column_price = f'Price_MA_{price_ma_window}'
        overlay_name_price = f'{price_ma_window}일 이동평균'

        if overlay_column_price not in df_calc.columns:
            df_calc[overlay_column_price] = df_calc['Price'].rolling(window=price_ma_window).mean()
    else:
        overlay_column_price = 'Price_Trend'
        overlay_name_price = '주가 선형 추세선'

    # --- 2. 주가 그래프 생성 (최상단) ---
    st.markdown(f"### 📈 {ticker_symbol} 주가 및 보조선 분석")

    fig_price = go.Figure()

    # 종가 (Price)
    fig_price.add_trace(go.Scatter(
        x=df_calc.index, y=df_calc['Price'],
        mode='lines', name='종가 (Price)',
        line=dict(color='blue', width=1.5)
    ))

    # 보조선 (MA 또는 추세선)
    fig_price.add_trace(go.Scatter(
        x=df_calc.index, y=df_calc[overlay_column_price],
        mode='lines', name=overlay_name_price,
        line=dict(color='red', dash='dash', width=2)
    ))

    fig_price.update_layout(
        title=f"{ticker_symbol} 주가 추이",
        height=500,
        xaxis_title="날짜",
        yaxis_title="주가 (Price)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- 3. 위젯 재배치 (그래프 아래 - 화면 표시용) ---
    st.markdown("---")
    st.markdown("### 🛠️ 보조선 설정 (위 그래프에 적용)")

    col_config_bottom1, col_config_bottom2 = st.columns(2)

    with col_config_bottom1:
        st.selectbox(
            "**보조선 선택**",
            ["선형 추세선", "이평선 (이동평균선)"],
            key='price_overlay_key_visible',
        )

    if st.session_state.price_overlay_key_visible == "이평선 (이동평균선)":
        with col_config_bottom2:
            st.number_input(
                "**이평선 기간 (일)**",
                min_value=1,
                max_value=300,
                step=5,
                key='price_ma_window_key_visible',
                format="%d",
                help="차트에 표시할 이동평균선의 기간을 설정합니다."
            )
    else:
        with col_config_bottom2:
            st.markdown(" ")
# ==============================================================================
# 섹션 5: PER 기반 QQQ 동적 매매 시뮬레이터 (새로운 Tab)
# ==============================================================================
elif st.session_state.active_tab == "PER 기반 QQQ 동적 매매 시뮬레이터":

    st.header("📈 PER 기반 QQQ 동적 매매 시뮬레이터")
    st.info("이 시뮬레이터는 **선택된 빅테크 종목의 합산 PER**을 지표로 사용하여 **QQQ ETF**를 매수/매도하는 전략을 백테스팅합니다. (매매 대상: QQQ)")
    st.markdown("---")

    # --- 1. 시뮬레이션 설정 (사이드바 대신 메인 화면에 위치) ---
    st.subheader("🛠️ 시뮬레이션 설정")

    col_sim_1, col_sim_2, col_sim_3 = st.columns(3)

    with col_sim_1:
        # 초기 투자금 (USD)
        sim_init_inv = st.number_input("초기 투자금 (USD)", min_value=1000, value=10000, step=1000, key='sim_init_inv_5')

    with col_sim_2:
        # 초기 현금 풀 (USD)
        sim_init_cash = st.number_input("초기 현금 풀 (USD)", min_value=0, value=1000, step=100, key='sim_init_cash_5')

    with col_sim_3:
        # 정기 적립금 (USD)
        sim_reg_deposit = st.number_input("정기 적립금 (USD)", min_value=10, value=500, step=50, key='sim_reg_deposit_5')

    col_sim_4, col_sim_5 = st.columns(2)
    with col_sim_4:
        deposit_freq = st.selectbox("적립 주기", options=["매일", "매주", "매월"], index=2, key='deposit_freq_5')

    if deposit_freq == "매일":
        deposit_freq_days = 1
    elif deposit_freq == "매주":
        deposit_freq_days = 7
    else:
        deposit_freq_days = 30  # 매월

    # PER 분석 대상 종목 선택 (재무 분석 탭의 기본값 사용)
    selected_tickers = st.multiselect(
        "PER 계산 대상 종목 (지표로 사용):",
        options=DEFAULT_BIG_TECH_TICKERS,
        default=DEFAULT_BIG_TECH_TICKERS,
        key='selected_tickers_5',
        help="이 종목들의 평균 PER이 QQQ 매매 기준이 됩니다."
    )

    if not selected_tickers:
        st.warning("PER 분석 대상 종목을 선택해주세요.")
        st.stop()

    # --- 2. 데이터 로드 및 시뮬레이션 실행 ---
    avg_per_hist, hist_error = load_historical_per_and_qqq_data(
        selected_tickers,
        start_date=start_date_final,  # 사이드바의 기간 설정 사용
        end_date=end_date_final
    )

    if hist_error:
        st.error(f"시계열 데이터 로드 오류: {hist_error}")
        st.stop()
    elif avg_per_hist.empty or avg_per_hist['Avg_PER'].isnull().all():
        st.warning("PER 시계열 데이터를 가져올 수 없습니다. 기간 및 종목을 확인해주세요.")
        st.stop()

    try:
        sim_results = run_dynamic_per_simulation(
            avg_per_hist,
            sim_init_inv,
            sim_init_cash,
            sim_reg_deposit,
            deposit_freq_days
        )
    except Exception as e:
        st.error(f"시뮬레이션 실행 중 오류 발생: {e}")
        st.stop()

    # --- 3. 최종 요약 결과 ---
    st.markdown("---")
    st.subheader("결과 요약 및 그래프")

    final_value = sim_results['Portfolio_Value'].iloc[-1]
    final_investment = sim_results['Total_Investment'].iloc[-1]
    final_return = final_value - final_investment
    final_ror = (final_return / final_investment) * 100 if final_investment > 0 else 0

    final_stock_value = sim_results['Stock_Value'].iloc[-1]
    final_cash_value = sim_results['Cash_Pool'].iloc[-1]
    final_stock_ratio = (final_stock_value / final_value) * 100 if final_value > 0 else 0

    # QQQ 단순 적립식 보유 성과 계산 (Buy & Hold)
    qqq_prices = sim_results['QQQ_Price']
    qqq_start_price = qqq_prices.iloc[0]

    buy_and_hold_shares = sim_init_inv / qqq_start_price if qqq_start_price > 0 else 0
    buy_and_hold_investment = sim_init_inv  # 초기 투자금 (주식 매수에 사용)
    last_deposit_date_bh = sim_results.index[0]

    for i in range(1, len(sim_results.index)):
        current_date = sim_results.index[i]
        deposit_added = 0
        if (current_date - last_deposit_date_bh).days >= deposit_freq_days:
            deposit_added = sim_reg_deposit
            last_deposit_date_bh = current_date

        if qqq_prices.iloc[i] > 0 and deposit_added > 0:
            buy_and_hold_shares += deposit_added / qqq_prices.iloc[i]

        buy_and_hold_investment += deposit_added

    qqq_hold_value = buy_and_hold_shares * qqq_prices.iloc[-1] + sim_init_cash
    qqq_hold_total_invest = buy_and_hold_investment + sim_init_cash

    qqq_return = qqq_hold_value - qqq_hold_total_invest
    qqq_ror = (qqq_return / qqq_hold_total_invest) * 100 if qqq_hold_total_invest > 0 else 0

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)  # col_res5 제거됨
    col_res1.metric("최종 포트폴리오 가치", f"${final_value:,.0f}")
    col_res2.metric("총 투자 원금", f"${final_investment:,.0f}")
    col_res3.metric("총 수익", f"${final_return:,.0f}", delta=f"{final_ror:,.2f}%")
    col_res4.metric("QQQ Buy & Hold 최종 가치", f"${qqq_hold_value:,.0f}", delta=f"{qqq_ror:,.2f}%")

    st.markdown("---")

    # --- 4. 시뮬레이션 결과 그래프 (QQQ, 포트폴리오 가치, PER) ---

    # Hover 정보를 위한 주식/현금 비율 계산
    sim_results['Stock_Ratio'] = (sim_results['Stock_Value'] / sim_results['Portfolio_Value']) * 100
    sim_results['Cash_Ratio'] = (sim_results['Cash_Pool'] / sim_results['Portfolio_Value']) * 100
    sim_results = sim_results.fillna({'Stock_Ratio': 0, 'Cash_Ratio': 0})

    fig_sim = go.Figure()

    # QQQ와 Portfolio_Value의 스케일을 맞추기 위한 정규화
    merged_results = sim_results[['Portfolio_Value', 'QQQ_Price']].dropna()

    if not merged_results.empty:
        ps_min = merged_results['Portfolio_Value'].min()
        ps_max = merged_results['Portfolio_Value'].max()
        qqq_min = merged_results['QQQ_Price'].min()
        qqq_max = merged_results['QQQ_Price'].max()


        def normalize_price(price, min_val, max_val, target_min, target_max):
            if max_val == min_val or target_max == target_min: return target_min
            return (price - min_val) / (max_val - min_val) * (target_max - target_min) + target_min


        target_min = ps_min * 0.95
        target_max = ps_max * 1.05

        # 정규화된 QQQ 가격
        normalized_qqq = merged_results['QQQ_Price'].apply(
            lambda x: normalize_price(x, qqq_min, qqq_max, target_min, target_max)
        )

        # 3. 평균 PER (우측 Y축)
        fig_sim.add_trace(go.Scatter(
            x=sim_results.index,
            y=sim_results['Avg_PER'],
            mode='lines',
            name='평균 PER',
            line=dict(color='blue', width=1, dash='dash'),
            opacity=0.7,
            yaxis='y2'
        ))

        # 1. 정규화된 QQQ (배경 그래프)
        fig_sim.add_trace(go.Scatter(
            x=merged_results.index,
            y=normalized_qqq,
            mode='lines',
            name='QQQ 종가 (정규화)',
            line=dict(color='gray', width=1),
            opacity=0.40,
            yaxis='y1'
        ))

    # 2. 포트폴리오 가치 (좌측 Y축)
    fig_sim.add_trace(go.Scatter(
        x=sim_results.index,
        y=sim_results['Portfolio_Value'],
        mode='lines',
        name='총 자산',
        line=dict(color='green', width=3),
        yaxis='y1',
        # Hovertemplate 및 customdata 설정 (주식/현금 비율 포함)
        customdata=sim_results[['Stock_Ratio', 'Cash_Ratio', 'Avg_PER', 'Stock_Value', 'Cash_Pool']],
        hovertemplate=(
                " $%{y:,.0f}<br>" +
                "<b>주식 비율:</b> %{customdata[0]:.2f}%"
        )
    ))

    # 3. 총 투자 원금 (좌측 Y축)
    fig_sim.add_trace(go.Scatter(
        x=sim_results.index,
        y=sim_results['Total_Investment'],
        mode='lines',
        name='총 투자 원금',
        line=dict(color='red', width=1.5, dash='dot'),
        yaxis='y1'
    ))

    # 4. PER 기준 가로선 추가 (우측 Y축)
    per_line_styles = {
        PER_CRITERIA_DYNAMIC['BUY_3X']: ('green', '30, 3x Buy'),
        PER_CRITERIA_DYNAMIC['BUY_2X']: ('darkgreen', '32, 2x Buy'),
        PER_CRITERIA_DYNAMIC['BUY_1X']: ('blue', '35, 1x Buy'),
        PER_CRITERIA_DYNAMIC['HOLD']: ('orange', '38, Hold'),
        PER_CRITERIA_DYNAMIC['SELL_15']: ('red', '41, 15% Sell'),
        PER_CRITERIA_DYNAMIC['SELL_30']: ('darkred', '45, 30% Sell')
    }

    per_levels_sorted = sorted(list(set(PER_CRITERIA_DYNAMIC.values())))

    for level in per_levels_sorted:
        if level in [PER_CRITERIA_DYNAMIC['BUY_3X'], PER_CRITERIA_DYNAMIC['BUY_2X'], PER_CRITERIA_DYNAMIC['BUY_1X'],
                     PER_CRITERIA_DYNAMIC['HOLD'], PER_CRITERIA_DYNAMIC['SELL_15'], PER_CRITERIA_DYNAMIC['SELL_30']]:
            color, label = per_line_styles.get(level, ('gray', f'{level:.1f}'))

            fig_sim.add_shape(
                type="line", xref="paper", yref="y2",
                x0=0, y0=level, x1=1, y1=level,
                line=dict(color=color, width=1, dash="dot"),
            )
            fig_sim.add_annotation(
                x=sim_results.index[-1], y=level, yref="y2",
                text=label.split(',')[0], showarrow=False,
                xanchor="left", yshift=5, font=dict(size=10, color=color)
            )

    fig_sim.update_layout(
        title="PER 기반 QQQ 동적 매매 전략 시뮬레이션 결과 ",
        height=600,
        xaxis_title="날짜",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, yanchor="top", xanchor="left"),

        yaxis=dict(
            title=dict(text="포트폴리오 가치 / 원금 / QQQ (USD)", font=dict(color="green")),
            side="left",
            showgrid=True,
            zeroline=False,
            tickformat="$,.0f"
        ),
        yaxis2=dict(
            title=dict(text="평균 PER (지표)", font=dict(color="blue")),
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickformat=".0f",
            range=[sim_results['Avg_PER'].min() * 0.9, sim_results['Avg_PER'].max() * 1.1]
        )
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 PER 기준 및 매매 로직")

    per_data_table = [
        ("< 30", "3배 매수", "정기 적립금의 3배 매수 + 현금 풀의 50% 재투자"),
        ("30 ~ < 32", "2배 매수", "정기 적립금의 2배 매수 + 현금 풀의 30% 재투자"),
        ("32 ~ < 35", "1배 매수", "정기 적립금의 1배 매수 + 현금 풀의 10% 재투자"),
        ("35 ~ < 38", "현금 보유 (0배)", "매매하지 않음. 정기 적립금을 Cash Pool에 적립"),
        ("38 ~ < 41", "15% 매도", "보유 주식의 15% 매도 + 정기 적립금을 Cash Pool에 적립"),
        ("41 ~ < 45", "30% 매도", "보유 주식의 30% 매도 + 정기 적립금을 Cash Pool에 적립"),
        (">= 45", "50% 매도", "보유 주식의 50% 매도 + 정기 적립금을 Cash Pool에 적립")
    ]

    df_per_table = pd.DataFrame(per_data_table, columns=["PER 구간", "권장 조치", "매매 로직"])
    st.table(df_per_table)




