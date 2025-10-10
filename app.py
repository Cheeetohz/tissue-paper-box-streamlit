# app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Tissue Paper Box — Portfolio Optimizer", layout="wide")

st.title("Tissue Paper Box Global Portfolio Optimizer")
st.markdown("""
Analyze and optimize a real-world multi-asset portfolio using Modern Portfolio Theory (MPT).  
This app uses live market data (NSE/US), Monte Carlo simulation, and SciPy optimization  
to build and evaluate an optimal portfolio based on the Sharpe ratio.
""")

# Sidebar Inputs
st.sidebar.header("Portfolio Settings")
tickers_input = st.sidebar.text_input(
    "Enter comma-separated tickers (e.g. RELIANCE.NS, INFY.NS, TCS.NS, GOLDBEES.NS)",
    "RELIANCE.NS, INFY.NS, TCS.NS, GOLDBEES.NS, TATAMOTORS.NS"
)
tickers = [t.strip() for t in tickers_input.split(",")]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
rf = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 3.0)
num_portfolios = st.sidebar.slider("Number of Random Portfolios", 2000, 20000, 5000, step=1000)

if st.sidebar.button("Run Optimization"):
    st.info("Fetching and validating data. Please wait a few seconds.")

    valid_data = {}
    summary_rows = []

    for t in tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if df.empty:
                df_recent = yf.download(t, start="2023-01-01", end=end_date, auto_adjust=True, progress=False)
                if not df_recent.empty:
                    st.warning(f"Limited data for {t}. Using data from {df_recent.index[0].strftime('%Y-%m-%d')} onward.")
                    df = df_recent
                else:
                    st.error(f"No valid data found for {t}. Skipping.")
                    summary_rows.append({'Ticker': t, 'Start': '-', 'End': '-', 'Total Days': 0, 'Status': 'No Data'})
                    continue

            df = df['Close'].asfreq('B')
            df = df.ffill().bfill()

            valid_data[t] = df
            summary_rows.append({
                'Ticker': t,
                'Start': df.index[0].strftime('%Y-%m-%d'),
                'End': df.index[-1].strftime('%Y-%m-%d'),
                'Total Days': len(df),
                'Status': 'OK'
            })

        except Exception as e:
            st.error(f"Error fetching {t}: {str(e)}")
            summary_rows.append({'Ticker': t, 'Start': '-', 'End': '-', 'Total Days': 0, 'Status': f'Error: {str(e)[:30]}'})
            continue

    if not valid_data:
        st.error("No valid data found for any tickers. Please check inputs and try again.")
        st.stop()

    data = pd.concat(valid_data.values(), axis=1)
    data.columns = valid_data.keys()

    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Data Fetch Summary")
    st.dataframe(summary_df, use_container_width=True)

    data = data.dropna(how='all')
    log_returns = np.log(data / data.shift(1)).dropna()
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252
    rf_rate = rf / 100
    num_assets = len(valid_data.keys())

    # Optimization Functions
    def portfolio_stats(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - rf_rate) / vol
        return ret, vol, sharpe

    def neg_sharpe(weights):
        return -portfolio_stats(weights)[2]

    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init_guess = np.repeat(1 / num_assets, num_assets)

    opt_sharpe = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    w_sharpe = opt_sharpe.x
    ret_sharpe, vol_sharpe, sr_sharpe = portfolio_stats(w_sharpe)

    weights_df = pd.DataFrame({'Asset': valid_data.keys(), 'Weight (%)': np.round(w_sharpe * 100, 2)})

    # Monte Carlo Simulation
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        port_ret = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_sharpe = (port_ret - rf_rate) / port_vol
        results[0, i] = port_ret
        results[1, i] = port_vol
        results[2, i] = port_sharpe
    results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])

    # Risk Metrics
    st.subheader("Risk Metrics")
    daily_returns = (log_returns * w_sharpe).sum(axis=1) if not log_returns.empty else pd.Series(dtype=float)

    if daily_returns.empty or daily_returns.isna().all():
        st.warning("Insufficient data to compute risk metrics. Try a shorter date range or different tickers.")
    else:
        sharpe_ratio = ((daily_returns.mean() - rf_rate / 252) / daily_returns.std()) * np.sqrt(252)
        downside = daily_returns[daily_returns < 0]
        sortino_ratio = ((daily_returns.mean() - rf_rate / 252) / downside.std()) * np.sqrt(252) if not downside.empty else np.nan

        try:
            var_99 = np.percentile(daily_returns.dropna(), 1)
            cvar_99 = daily_returns[daily_returns <= var_99].mean()
        except Exception:
            var_99, cvar_99 = np.nan, np.nan

        st.markdown(f"""
        **Expected Annual Return:** {ret_sharpe:.2%}  
        **Annual Volatility:** {vol_sharpe:.2%}  
        **Sharpe Ratio:** {sharpe_ratio:.2f}  
        **Sortino Ratio:** {sortino_ratio:.2f}  
        **99% VaR:** {var_99:.2%}  
        **99% CVaR:** {cvar_99:.2%}  
        """)

    # Efficient Frontier Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', alpha=0.6)
    ax.scatter(vol_sharpe, ret_sharpe, marker='*', color='red', s=250, label='Max Sharpe Portfolio')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Expected Return')
    ax.legend()
    fig.colorbar(scatter, label='Sharpe Ratio')
    st.pyplot(fig)

    # Download Button
    csv = weights_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Optimized Portfolio Weights (CSV)",
        data=csv,
        file_name='optimized_portfolio.csv',
        mime='text/csv'
    )

    st.success("Optimization complete. You can re-run with different tickers or date ranges.")

    # Portfolio Growth Simulation
    st.subheader("Portfolio Growth Simulation (₹10 Lakh Initial Investment)")
    initial_investment = 10_00_000

    cumulative_portfolio = pd.Series(dtype=float)
    cumulative_nifty = pd.Series(dtype=float)

    if not daily_returns.empty:
        cumulative_portfolio = (1 + daily_returns).cumprod() * initial_investment

    try:
        nifty = yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
        if not nifty.empty:
            nifty_returns = np.log(nifty / nifty.shift(1)).dropna()
            cumulative_nifty = (1 + nifty_returns).cumprod() * initial_investment
            if not cumulative_portfolio.empty:
                cumulative_nifty = cumulative_nifty.reindex(cumulative_portfolio.index, method='ffill')
    except Exception:
        st.warning("Could not fetch NIFTY 50 data for benchmark comparison.")

    # Convert final values to scalars safely
    # Convert final values to scalars safely
    final_portfolio_value = float(cumulative_portfolio.values[-1]) if not cumulative_portfolio.empty else np.nan
    final_nifty_value = float(cumulative_nifty.values[-1]) if not cumulative_nifty.empty else np.nan


    if not pd.isna(final_portfolio_value):
        st.write(f"If you had invested ₹10,00,000 on {start_date.strftime('%d %b %Y')}:")
        if not pd.isna(final_nifty_value):
            st.markdown(f"""
            - Optimized Portfolio: ₹{final_portfolio_value:,.0f}  
            - NIFTY 50 Benchmark: ₹{final_nifty_value:,.0f}
            """)
        else:
            st.markdown(f"- Optimized Portfolio: ₹{final_portfolio_value:,.0f}")

    # Plot growth
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    if not cumulative_portfolio.empty:
        ax2.plot(cumulative_portfolio.index, cumulative_portfolio, label='Optimized Portfolio', color='teal', linewidth=2)
    if not cumulative_nifty.empty:
        ax2.plot(cumulative_nifty.index, cumulative_nifty, label='NIFTY 50', color='orange', linestyle='--', linewidth=2)

    ax2.set_title("Portfolio vs Benchmark Growth")
    ax2.set_ylabel("Portfolio Value (₹)")
    ax2.legend()
    st.pyplot(fig2)

    st.success("Simulation complete. This represents real-world portfolio performance.")
