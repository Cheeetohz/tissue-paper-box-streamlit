Tissue Paper Box — Streamlit Portfolio Optimizer

Interactive web app to optimize a **real world portfolio** using Modern Portfolio Theory (MPT).  
Built with Python, Streamlit, SciPy, and live market data via Yahoo Finance.

Live Demo
*(deploy it on Streamlit Cloud and paste your link here)*

Features
- Fetches real NSE/US stock & ETF data
- Runs Monte Carlo simulation (10k+ portfolios)
- SciPy optimization for Max Sharpe Ratio
- Calculates Sharpe, Sortino, VaR, CVaR
- Downloadable CSV of optimal weights

Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
