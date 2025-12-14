import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from datetime import date, timedelta


# --- 1. 데이터 로드 및 캐싱 함수 ---

@st.cache_data
def load_ticker_info(ticker):
    """티커 정보를 로드합니다 (EPS, 회사 이름)."""
    try:
        data = yf.Ticker(ticker)
        info = data.info

        # EPS (Trailing EPS 선호, 없으면 Forward EPS 시도)
        eps = info.get('trailingEps')
        if eps is None or eps == 0:
            eps = info.get('forwardEps')

        # PER 계산용 데이터프레임 구조
        per_info = {
            'EPS': eps if eps else 0,
            'CompanyName': info.get('longName', ticker),
        }
        return per_info, None
    except Exception:
        return None, "Ticker information could not be loaded."


@st.cache_data
def load_historical_data(ticker, start_date, end_date):
    """yfinance에서 주가 데이터를 로드합니다."""
    if start_date == 'max':
        start_date = None

    try:
        hist = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if hist.empty:
            return None, "해당 기간의 주가 데이터를 가져올 수 없습니다."
        return hist, None
    except Exception as e:
        return None, f"데이터 로드 중 오류가 발생했습니다: {e}"


@st.cache_data
def load_big_tech_data(tickers):
    """요청된 빅테크 종목의 재무 정보를 로드합니다."""

    # 딕셔너리 리스트로 데이터를 수집
    data_list = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info

            market_cap = info.get('marketCap', np.nan)
            trailing_pe = info.get('trailingPE', np.nan)

            # TTM Net Income (순이익)을 직접적으로 가져오기 어려우므로,
            # 시가총액(Market Cap)과 TTM PER을 사용하여 역산
            # Net Income = Market Cap / PER
            net_income = market_cap / trailing_pe if market_cap and trailing_pe and trailing_pe > 0 else np.nan

            data_list.append({
                'Ticker': ticker,
                'MarketCap': market_cap,
                'TrailingPE': trailing_pe,
                'NetIncome': net_income,
            })
        except Exception:
            data_list.append({
                'Ticker': ticker,
                'MarketCap': np.nan,
                'TrailingPE': np.nan,
                'NetIncome': np.nan,
            })

    return pd.DataFrame(data_list)


@st.cache_data
def format_value(val):
    """숫자를 T (조), B (십억) 단위로 포매팅합니다."""
    if pd.isna(val) or val == 0:
        return "-"
    # 절대값 기준으로 비교
    if abs(val) >= 1e12:
        return f"{val / 1e12:,.2f}T"
    elif abs(val) >= 1e9:
        return f"{val / 1e9:,.2f}B"
    return f"{val:,.2f}"


# --- 2. PER 및 보조 지표 계산 함수 ---

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

        # 잔차(Residuals) 계산: 실제 PER - 추세선 PER
        valid_per_data = data.loc[per_data_for_calc.index].copy()
        data['PER_Residual'] = np.nan
        data.loc[valid_per_data.index, 'PER_Residual'] = valid_per_data['PER'] - valid_per_data['PER_Trend']

        # 잔차의 표준편차 계산 (PER_SD)
        per_sd = data['PER_Residual'].std()

        data['PER_SD'] = per_sd

        # 현재 시점 데이터 추출
        if per_sd > 0 and not data.empty:
            current_per = data['PER'].iloc[-1]
            current_trend = data['PER_Trend'].iloc[-1]

            # Z-Score 계산 (Z_PER)
            z_score = (current_per - current_trend) / per_sd

            # 매력도 점수 계산 (100점에서 시작, 0점으로 클리핑)
            score = 100 * (1 - (z_score + 2) / 4)

            # 0~100 범위로 클리핑
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


# --- 3. Streamlit UI 및 레이아웃 설정 ---

st.set_page_config(layout="wide", page_title="주식 분석 앱")


# --- 기간 설정 (기본값) ---
TODAY = date.today()
ONE_YEAR_AGO = TODAY - timedelta(days=365)

# --- 사이드바: 기본 설정 ---
st.sidebar.header("⚙️ 기본 설정")

# 3-1. 티커 입력 (기본값 QQQ)
ticker_symbol = st.sidebar.text_input(
    "**주식 티커를 입력하세요:**",
    value="NVDA",
    help="이 탭에 표시되는 티커는 탭 2, 3, 4의 분석 대상이 됩니다."
).upper()

# 3-2. 기간 선택 드롭다운
period_options = {
    "1개월": 30, "3개월": 90, "6개월": 180, "1년": 365, "2년": 730, "5년": 1825, "YTD (연초 대비)": 'ytd', "최대 기간": 'max'
}
selected_period_name = st.sidebar.selectbox(
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

start_date_input = st.sidebar.date_input(
    "**시작 날짜 (직접 입력):**",
    value=start_date_default,
    min_value=date(1900, 1, 1),
    max_value=TODAY
)

end_date_input = st.sidebar.date_input(
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
info, info_error = load_ticker_info(ticker_symbol)

if info_error:
    st.error(f"티커 정보를 가져오는 데 실패했습니다: {info_error}")
    st.stop()

st.subheader(f"🚀 {info['CompanyName']} ({ticker_symbol}) 분석")

hist_data, data_error = load_historical_data(
    ticker_symbol,
    start_date=start_date_final,
    end_date=end_date_final
)

if data_error:
    st.error(f"데이터 로드 오류: {data_error}")
    st.stop()

# 최종 데이터 계산
df_calc = calculate_per_and_indicators(hist_data, info['EPS'])

# --- 4. 2x2 네모 박스 메뉴 구현 ---

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "재무 분석"  # 초기 선택 메뉴

menu_options = [
    "재무 분석",
    "적립 모드 (DCA)",
    "PER 그래프 분석",
    "주가 및 이동평균선"
]
cols = st.columns(2)
cols.extend(st.columns(2))  # 2x2 레이아웃을 위한 4개의 컬럼 리스트



for i, option in enumerate(menu_options):
    with cols[i]:
        # 현재 활성화된 탭이면 primary(강조), 아니면 secondary(일반) 스타일 적용
        is_active = (st.session_state.active_tab == option)
        # Streamlit 1.25+ 버전부터 'type' 파라미터로 버튼 스타일 지정 가능
        button_type = "primary" if is_active else "secondary"

        if st.button(
                option,
                key=f"tab_button_{i}",
                use_container_width=True,
                type=button_type
        ):
            st.session_state.active_tab = option
            st.rerun()  # 버튼 클릭 시 화면 새로고침하여 내용 변경

st.markdown("---")

# --- 5. 선택된 메뉴에 따른 내용 표시 (조건문으로 구현) ---

# ==============================================================================
# 섹션 1: 재무 분석 (빅테크)
# ==============================================================================
if st.session_state.active_tab == "재무 분석":

    BIG_TECH_TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA']

    # 데이터 로드 (캐싱된 함수 사용)
    tech_df_raw = load_big_tech_data(BIG_TECH_TICKERS)


    # 1. 체크박스(선택) 컬럼을 추가하여 Data Editor에 사용

    # Session State 초기화 (모두 True로 설정)
    if 'tech_select_state' not in st.session_state:
        # 모든 종목을 초기 True로 설정
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


# ==============================================================================
# 섹션 2: 적립 모드 (DCA 시뮬레이션)
# ==============================================================================
elif st.session_state.active_tab == "적립 모드 (DCA)":

    # --- 1. Session State 초기화 (위젯 값이 없을 때만 실행) ---
    if 'dca_amount' not in st.session_state:
        st.session_state.dca_amount = 10.0
    if 'dca_freq' not in st.session_state:
        st.session_state.dca_freq = "매일"

    # --- 2. 시뮬레이션 계산 (그래프를 그리기 위한 사전 계산) ---
    # Session State에서 현재 값을 가져와 계산에 사용
    deposit_amount = st.session_state.dca_amount
    deposit_frequency = st.session_state.dca_freq

    dca_df = df_calc.copy()
    dca_df['DayOfYear'] = dca_df.index.dayofyear
    dca_df['WeekOfYear'] = dca_df.index.isocalendar().week.astype(int)
    dca_df['Month'] = dca_df.index.month
    if deposit_frequency == "매일":
        # 매일은 전체 인덱스를 사용
        invest_dates = dca_df.index
    elif deposit_frequency == "매주":
        # 각 WeekOfYear 그룹의 첫 번째 항목 인덱스(날짜)를 사용
        # .index 대신 .first().index를 사용하면 Series의 인덱스를 가져와서 오류가 발생함.
        # dca_df.groupby('WeekOfYear')['Price'].first().index는 WeekOfYear이므로,
        # dca_df.groupby('WeekOfYear')['Price'].idxmin() 또는 idxmax()를 사용해야 하지만,
        # 가장 간단한 방법은 dca_df의 인덱스를 리스트로 가져오는 것입니다.
        invest_dates = dca_df.groupby('WeekOfYear')['Price'].head(1).index
    elif deposit_frequency == "매월":
        # 각 Month 그룹의 첫 번째 항목 인덱스(날짜)를 사용
        # 마찬가지로 dca_df의 인덱스를 리스트로 가져옵니다.
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
        # value 인수를 생략하고 key만 사용하면 Session State의 값이 자동으로 사용됨
        st.number_input(
            "**적립 금액 (USD)**",
            min_value=1.0,
            step=1.0,
            format="%.2f",
            key='dca_amount',
            help="매번 투자할 금액을 입력합니다."
        )

    with col_dca_config2:
        # index도 현재 Session State 값을 기준으로 계산하여 사용
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
# 섹션 3: PER 그래프 분석
# ==============================================================================
elif st.session_state.active_tab == "PER 그래프 분석":

    per_data_filtered = df_calc[df_calc['PER'] != np.inf]

    # PER 데이터를 분석할 수 있을 때만 로직 실행
    if per_data_filtered.empty:
        st.warning("PER 계산을 위한 유효한 EPS 데이터가 없거나, EPS가 0 이하입니다. ETF가 아닌 실제 기업의 Ticker를 입력해주세요.")
    else:
        # --- 그래프 생성 (PER 및 선형 추세선) (최상단) ---

        overlay_column = 'PER_Trend'
        overlay_name = 'PER 선형 추세선'

        # PER_SD를 사용하여 Z-Score와 Score를 계산하여 hover_data로 준비
        hover_data = per_data_filtered.copy()

        # 각 시점의 Z-Score와 매력도 점수를 재계산 (NaN 처리 포함)
        hover_data['Calculated_Z_Score'] = (hover_data['PER'] - hover_data['PER_Trend']) / hover_data['PER_SD']
        hover_data['Calculated_Score'] = 100 * (1 - (hover_data['Calculated_Z_Score'] + 2) / 4)

        # 0~100 범위로 클리핑
        hover_data['Display_Score'] = hover_data['Calculated_Score'].apply(lambda s: max(0, min(100, s))).round(
            0)  # 점수는 소수점 없이 표시
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
            hovermode="x unified",  # x축 기준으로 통합하여 표시
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
# 섹션 4: 주가 그래프 및 이동평균선/추세선
# ==============================================================================
elif st.session_state.active_tab == "주가 및 이동평균선":

    # --- 1. Session State 초기화 및 값 로드 (그래프 계산에 사용) ---
    if 'price_overlay_key_visible' not in st.session_state:
        st.session_state.price_overlay_key_visible = "이평선 (이동평균선)"
    if 'price_ma_window_key_visible' not in st.session_state:
        st.session_state.price_ma_window_key_visible = 20

    price_overlay_choice = st.session_state.price_overlay_key_visible
    price_ma_window = st.session_state.price_ma_window_key_visible

    # 이평선을 선택했을 경우 윈도우 계산
    if price_overlay_choice == "이평선 (이동평균선)":
        # price_ma_window는 이미 int 타입이므로 형 변환 불필요 (Session State에서 가져옴)

        overlay_column_price = f'Price_MA_{price_ma_window}'
        overlay_name_price = f'{price_ma_window}일 이동평균'

        # 사용자가 선택한 기간이 기존 MA 목록에 없거나, 재계산이 필요한 경우
        if overlay_column_price not in df_calc.columns:
            # df_calc는 이미 위에서 계산되어 있다고 가정
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
        # **오류 수정:** index를 제거하고 key만 사용하여 Session State에 동기화
        st.selectbox(
            "**보조선 선택**",
            ["선형 추세선", "이평선 (이동평균선)"],
            key='price_overlay_key_visible',  # Session State에 직접 저장
        )

    if st.session_state.price_overlay_key_visible == "이평선 (이동평균선)":
        with col_config_bottom2:
            # **오류 수정:** value를 제거하고 key만 사용하여 Session State에 동기화
            st.number_input(
                "**이평선 기간 (일)**",
                min_value=1,
                max_value=300,
                step=5,
                key='price_ma_window_key_visible',  # Session State에 직접 저장
                format="%d",
                help="차트에 표시할 이동평균선의 기간을 설정합니다."
            )
    else:
        # 이평선이 아닐 경우 빈 공간을 채워 레이아웃을 유지합니다.
        with col_config_bottom2:
            st.markdown(" ") # 빈 줄


