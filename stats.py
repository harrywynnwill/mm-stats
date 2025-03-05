import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. Read CSV and Clean Data
# -----------------------------------------------------------------------------
df = pd.read_csv('trades.csv')
df.columns = df.columns.str.strip()

# Clean 'Profit' column so negative values aren't lost
df['Profit'] = df['Profit'].astype(str).str.replace(r'\s+', '', regex=True)
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)


# Debug check
print("=== DataFrame Head (first 10 rows) ===")
print(df.head(10))
print("\n=== 'Profit' Descriptive Stats After Cleaning ===")
print(df['Profit'].describe())

# -----------------------------------------------------------------------------
# 2. Set Initial Capital
# -----------------------------------------------------------------------------
initial_capital = 40000.0  # Adjust as needed

# -----------------------------------------------------------------------------
# 3. Sharpe Ratio Function
# -----------------------------------------------------------------------------
def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate the annualized Sharpe Ratio using percentage returns.
    - returns: Pandas Series or NumPy array of per-trade returns in decimal (e.g., 0.01 for 1%).
    - risk_free_rate: Annual risk-free rate. Default = 0.02 (2%).
    - periods_per_year: Trading periods (or trades) per year. Default = 252 (daily).
    """
    returns = np.array(returns.dropna())
    
    # Check for edge cases
    if len(returns) < 2 or np.all(returns == returns[0]):
        return np.nan
    
    avg_return = np.mean(returns)
    std_dev = np.std(returns, ddof=1)
    rf_per_period = risk_free_rate / periods_per_year
    
    if std_dev > 0:
        sharpe_per_period = (avg_return - rf_per_period) / std_dev
    else:
        sharpe_per_period = np.nan
    
    return sharpe_per_period * np.sqrt(periods_per_year)

# -----------------------------------------------------------------------------
# 4. Maximum Drawdown Function
# -----------------------------------------------------------------------------
def calculate_max_drawdown(equity_curve, debug=False):
    """
    Calculate the maximum drawdown from an equity curve.
    Returns drawdown in percentage (0% to 100%).
    """
    equity_curve = np.array(equity_curve)
    equity_curve = equity_curve[~np.isnan(equity_curve)]
    
    if len(equity_curve) == 0:
        return np.nan
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    
    if debug and len(equity_curve) > 0:
        print("\n=== Debug: First 10 Points of Equity Curve ===")
        for i in range(min(10, len(equity_curve))):
            print(f"Index {i} - Equity: {equity_curve[i]}, "
                  f"Run Max: {running_max[i]}, "
                  f"Drawdown: {drawdowns[i]}")
    
    return np.max(drawdowns) * 100

# -----------------------------------------------------------------------------
# 5. Main Calculation
# -----------------------------------------------------------------------------
df['Entry Time'] = pd.to_datetime(df['Time'].str.strip(), errors='coerce')
df['Exit Time'] = pd.to_datetime(df['Time.1'].str.strip(), errors='coerce')
df['Holding Period (Days)'] = (df['Exit Time'] - df['Entry Time']).dt.total_seconds() / (24 * 3600)

if not df['Exit Time'].dropna().empty and not df['Entry Time'].dropna().empty:
    total_days = (df['Exit Time'].max() - df['Entry Time'].min()).days
else:
    total_days = 0

num_trades = len(df)
trades_per_year = num_trades / (total_days / 365.25) if total_days > 0 else 252

profits = df['Profit']
returns = profits / initial_capital
cumulative_pnl = initial_capital + profits.cumsum()
total_pnl = profits.sum()

# Debug: Check the first 10 lines of the equity curve
print("\n=== Equity Curve (first 10) ===")
print(cumulative_pnl.head(10).tolist())

# Calculate metrics
sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=trades_per_year)
max_drawdown = calculate_max_drawdown(cumulative_pnl, debug=True)

# -----------------------------------------------------------------------------
# 6. Print Results
# -----------------------------------------------------------------------------
print("\n=== RESULTS ===")
print(f"Initial Capital: ${initial_capital:.2f}")
print(f"Number of Trades: {num_trades}")
print(f"Total Days: {total_days}")
print(f"Estimated Trades per Year: {trades_per_year:.2f}")
print(f"Total PnL: ${total_pnl:.2f}")
if not cumulative_pnl.empty:
    print(f"Final Equity: ${cumulative_pnl.iloc[-1]:.2f}")
print(f"Average Return per Trade: {returns.mean()*100:.2f}%")
print(f"Standard Deviation of Returns: {returns.std(ddof=1)*100:.2f}%")
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")

# -----------------------------------------------------------------------------
# 7. Calculate & Print Annualized Return
# -----------------------------------------------------------------------------
if total_days > 0 and not cumulative_pnl.empty:
    final_equity = cumulative_pnl.iloc[-1]
    total_return = final_equity / initial_capital - 1  # e.g. 0.25 for a 25% total gain
    # Convert total return to annualized (geometric) return
    annual_return = (1 + total_return) ** (365 / total_days) - 1
    annual_return_pct = annual_return * 100
    print(f"Annualized Return (Geometric): {annual_return_pct:.2f}%")
else:
    print("Annualized Return: N/A (no valid days in dataset)")

# -----------------------------------------------------------------------------
# Optional: Trade Statistics
# -----------------------------------------------------------------------------
print("\nTrade Statistics (USD):")
print(df['Profit'].describe())
print("\nTrade Statistics (Returns %):")
print((returns * 100).describe())
