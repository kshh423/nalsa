import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.stats import linregress
import numpy as np

# --- 1. 앱 설정 및 제목 ---
st.set_page_config(layout="wide")
st.title("📈 주식 티커별 일별 PER(Price-to-Earnings Ratio) 그래프")
st.markdown("티커(예: **AAPL**, **MSFT**, **005930.KS** 등)를 입력하고 기간을 선택하세요.")

# --- 2. 사용자 입력 받기 (티커 및 기간) ---
col1, col2 = st.columns([1, 1])

with col1:
    ticker_symbol = st.text_input(
        "**주식 티커를 입력하세요:**",
        value="005930.KS", # 삼성전자 (Korean Stock)
        help="예: AAPL (Apple), 005930.KS (삼성전자)"
    ).upper()

with col2:
    # yfinance가 지원하는 기간 옵션
    period_options = {
        "1년": "1y", "3개월": "3mo", "6개월": "6mo",
        "YTD (연초 대비)": "ytd", "2년": "2y", "5년": "5y",
        "최대 기간": "max"
    }
    selected_period_name = st.selectbox(
        "**조회 기간을 선택하세요:**",
        list(period_options.keys()),
        index=0
    )
    period = period_options[selected_period_name]


# --- 3. 데이터 로드 및 처리 ---

@st.cache_data
def load_data(ticker, period):
    """yfinance에서 주식 데이터를 로드하고 PER을 계산합니다."""
    try:
        # 주가 및 배당금 데이터 가져오기
        ticker_data = yf.Ticker(ticker)
        
        # 1. 주가 데이터 (Adj Close 사용)
        hist = ticker_data.history(period=period)
        if hist.empty:
            return None, "주가 데이터를 가져올 수 없습니다. 티커를 확인해 주세요."
        
        # 2. 재무 정보 (EPS를 찾기 위해)
        # yfinance는 일별 EPS 데이터를 제공하지 않으므로, 최근 4분기 EPS (Trailing EPS)를 사용합니다.
        # yfinance의 info 객체에서 'trailingEps' 또는 'forwardEps'를 사용하거나, 
        # TTM EPS를 계산하기 위해 quarterly_financials를 사용할 수 있지만, 간단하게 'trailingEps'를 사용해봅니다.
        
        # Ticker info를 한 번에 가져옵니다.
        info = ticker_data.info
        
        # Trailing EPS (최근 12개월 순이익/총 주식수)
        # 이 값은 일별로 변하지 않지만, PER 계산의 분모로 사용합니다.
        # 주의: 이 값은 yfinance가 제공하는 '가장 최근' TTM EPS이며, 주가 히스토리의 모든 날짜에 동일하게 적용됩니다. 
        # 실제로는 EPS도 분기마다 업데이트되므로, 그래프는 단순화된 버전임을 명시해야 합니다.
        
        # 'trailingEps'가 없으면 'forwardEps'를 시도합니다.
        eps = info.get('trailingEps') 
        if eps is None or eps == 0:
            eps = info.get('forwardEps')
        
        if eps is None or eps == 0:
            return None, "PER 계산을 위한 EPS (주당순이익) 데이터를 찾을 수 없습니다."

        # 3. 데이터프레임에 주가 및 EPS 추가
        df = hist.copy()
        df['Price'] = df['Close'] # 종가를 사용
        df['EPS'] = eps
        
        # 4. PER 계산 (PER = Price / EPS)
        # EPS가 0이면 무한대가 되므로, 0인 경우에 대한 처리가 필요합니다.
        df['PER'] = np.where(df['EPS'] > 0, df['Price'] / df['EPS'], np.inf)
        
        return df, None
    
    except Exception as e:
        return None, f"데이터를 로드하는 중 오류가 발생했습니다: {e}"

# 사용자가 티커를 입력하고 버튼을 눌렀을 때 실행
if st.button("📊 데이터 조회 및 그래프 그리기"):
    
    # 로딩 스피너 표시
    with st.spinner(f"**{ticker_symbol}**의 데이터를 로드하고 PER을 계산 중입니다..."):
        data_df, error_message = load_data(ticker_symbol, period)
    
    if error_message:
        st.error(f"⚠️ 오류 발생: {error_message}")
    elif data_df is not None and not data_df.empty:
        
        st.success(f"**{ticker_symbol}**의 PER 데이터 로드 완료. (기간: {selected_period_name})")
        
        # --- 4. PER 그래프 생성 (Plotly) ---
        
        # 무한대 PER 제거 (EPS가 0이거나 음수인 경우)
        per_data_for_plot = data_df[data_df['PER'] != np.inf]
        
        if per_data_for_plot.empty:
            st.warning("계산 가능한 PER 데이터가 없습니다. EPS가 0 이하일 수 있습니다.")
        else:
            
            # --- 그래프 그리기 ---
            fig = px.line(
                per_data_for_plot, 
                x=per_data_for_plot.index, 
                y='PER',
                title=f'{ticker_symbol} 일별 PER 추이 (EPS: {per_data_for_plot["EPS"].iloc[-1]:.2f} 기준)',
                labels={'x': '날짜', 'PER': 'PER (주가수익비율)'},
                template="plotly_white"
            )
            
            # 이동평균선 추가 (선택 사항)
            window = 20 # 20일 이동평균선
            per_data_for_plot['PER_MA'] = per_data_for_plot['PER'].rolling(window=window).mean()
            
            fig.add_scatter(
                x=per_data_for_plot.index, 
                y=per_data_for_plot['PER_MA'], 
                mode='lines', 
                name=f'{window}일 PER 이동평균',
                line=dict(color='red', dash='dot')
            )

            # --- 추세선 추가 (선택 사항) ---
            # 선형 회귀를 통해 간단한 추세선을 그립니다.
            
            # x축 데이터를 0부터 시작하는 숫자로 변환
            x_values = np.arange(len(per_data_for_plot)) 
            slope, intercept, r_value, p_value, std_err = linregress(x_values, per_data_for_plot['PER'])
            
            # 추세선 데이터 생성
            per_data_for_plot['Trendline'] = intercept + slope * x_values

            fig.add_scatter(
                x=per_data_for_plot.index, 
                y=per_data_for_plot['Trendline'], 
                mode='lines', 
                name='선형 추세선',
                line=dict(color='gray', dash='longdash')
            )
            
            # 레이아웃 설정
            fig.update_layout(
                xaxis_title="날짜",
                yaxis_title="PER",
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Streamlit에 그래프 표시
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 5. 데이터 요약 및 주의 사항 ---
            st.subheader("📝 데이터 요약 및 참고 사항")
            
            # 현재 PER
            current_per = per_data_for_plot['PER'].iloc[-1]
            st.markdown(f"* **최근 영업일 기준 PER:** **{current_per:.2f}**")

            st.info("""
            **⚠️ 중요 참고 사항:**
            * **yfinance의 EPS 한계:** 이 그래프는 yfinance에서 제공하는 **가장 최근의 TTM (Trailing Twelve Months) EPS**를 사용합니다. 이 EPS 값은 주가 조회 기간 동안 **변하지 않고 고정**됩니다.
            * **실제 PER:** 실제 증권사나 금융 웹사이트에서 제공하는 PER 그래프는 분기별 EPS 업데이트를 반영하여 계단식으로 변동합니다. 따라서 이 그래프는 **'고정된 EPS를 가정했을 때의 주가 변동에 따른 PER 추이'**를 보여주는 단순화된 모델입니다.
            * **PER 계산:** $\\text{PER} = \\frac{\\text{주가 (Price)}}{\\text{주당순이익 (EPS)}}$
            """)
            
            # 원본 데이터 표시 (선택 사항)
            if st.checkbox("원본 데이터 보기"):
                st.dataframe(per_data_for_plot[['Price', 'EPS', 'PER', 'PER_MA', 'Trendline']].tail(10))

    else:
        st.warning("👆 위에 티커를 입력하고 '데이터 조회 및 그래프 그리기' 버튼을 눌러주세요.")
