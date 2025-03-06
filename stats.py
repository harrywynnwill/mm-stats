import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest(trades_csv, initial_capital=40000.0, trade_multiplier=1.0):
    """
    Loads trades from CSV, scales them by trade_multiplier, 
    and calculates performance metrics using the specified initial_capital.
    Generates weekly PnL and drawdown charts (PNG files) 
    and logs the weekly stats in both the console and a CSV.
    """
    # -----------------------------------------------------------------------------
    # 1. Read CSV and Clean Data
    # -----------------------------------------------------------------------------
    df = pd.read_csv(trades_csv)
    df.columns = df.columns.str.strip()
    
    # Clean 'Profit' column so negative values aren't lost
    df['Profit'] = df['Profit'].astype(str).str.replace(r'\s+', '', regex=True)
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
    
    # Scale the trades by the chosen multiplier
    df['Profit'] *= trade_multiplier

    # Parse datetimes
    df['Entry Time'] = pd.to_datetime(df['Time'].str.strip(), errors='coerce')
    df['Exit Time'] = pd.to_datetime(df['Time.1'].str.strip(), errors='coerce')
    
    # -----------------------------------------------------------------------------
    # 2. (Optional) Filter by Product
    # -----------------------------------------------------------------------------
    products_to_keep = ['EURUSD', 'GOLD', 'USDJPY', '#Japan225', '#USNDAQ100']
    df = df[df['Symbol'].isin(products_to_keep)]
    
    # -----------------------------------------------------------------------------
    # 3. Time Period and Products
    # -----------------------------------------------------------------------------
    start_date = df['Entry Time'].min()
    end_date = df['Exit Time'].max()

    print("=== Time Period for Trades ===")
    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")

    unique_symbols = df['Symbol'].unique()
    print("\n=== Products in Data ===")
    print(unique_symbols)
    
    # -----------------------------------------------------------------------------
    # 4. Holding Period, Total Days, etc.
    # -----------------------------------------------------------------------------
    df['Holding Period (Days)'] = (df['Exit Time'] - df['Entry Time']).dt.total_seconds() / (24 * 3600)
    
    if not df['Exit Time'].dropna().empty and not df['Entry Time'].dropna().empty:
        total_days = (df['Exit Time'].max() - df['Entry Time'].min()).days
    else:
        total_days = 0
    
    num_trades = len(df)
    trades_per_year = num_trades / (total_days / 365.25) if total_days > 0 else 252
    
    # -----------------------------------------------------------------------------
    # 5. Sharpe Ratio Function
    # -----------------------------------------------------------------------------
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        returns_array = np.array(returns.dropna())
        if len(returns_array) < 2 or np.all(returns_array == returns_array[0]):
            return np.nan
        
        avg_return = np.mean(returns_array)
        std_dev = np.std(returns_array, ddof=1)
        rf_per_period = risk_free_rate / periods_per_year
        
        if std_dev > 0:
            sharpe_per_period = (avg_return - rf_per_period) / std_dev
        else:
            sharpe_per_period = np.nan
        
        return sharpe_per_period * np.sqrt(periods_per_year)
    
    # -----------------------------------------------------------------------------
    # 6. Maximum Drawdown Function
    # -----------------------------------------------------------------------------
    def calculate_max_drawdown(equity_curve, debug=False):
        equity_array = np.array(equity_curve.dropna())
        if len(equity_array) == 0:
            return np.nan
        
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array) / running_max
        
        if debug:
            print("\n=== Debug: First 10 Points of Equity Curve ===")
            for i in range(min(10, len(equity_array))):
                print(f"Index {i} - Equity: {equity_array[i]}, "
                      f"Running Max: {running_max[i]}, "
                      f"Drawdown: {drawdowns[i]}")
        
        return np.max(drawdowns) * 100
    
    # -----------------------------------------------------------------------------
    # 7. Compute the Equity Curve (trade-based)
    # -----------------------------------------------------------------------------
    profits = df['Profit']
    returns = profits / initial_capital
    cumulative_pnl = initial_capital + profits.cumsum()
    total_pnl = profits.sum()
    
    print("\n=== Equity Curve (first 10) ===")
    print(cumulative_pnl.head(10).tolist())
    
    # Calculate Sharpe & max drawdown on the full (trade-based) equity series
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=trades_per_year)
    max_drawdown_val = calculate_max_drawdown(cumulative_pnl, debug=True)
    
    # -----------------------------------------------------------------------------
    # 8. Annualized Return (Geometric)
    # -----------------------------------------------------------------------------
    if total_days > 0 and not cumulative_pnl.empty:
        final_equity = cumulative_pnl.iloc[-1]
        total_return = final_equity / initial_capital - 1
        annual_return = (1 + total_return) ** (365 / total_days) - 1
        annual_return_pct = annual_return * 100
    else:
        annual_return_pct = np.nan
    
    # -----------------------------------------------------------------------------
    # 9. Create a Weekly Time Series and Compute Weekly PnL & Drawdown
    # -----------------------------------------------------------------------------
    # Build a time series of equity based on exit times
    equity_ts = pd.Series(data=cumulative_pnl.values, index=df['Exit Time'])
    equity_ts = equity_ts.sort_index()
    
    # Resample to weekly frequency using the last known equity in each week
    weekly_equity = equity_ts.resample('W').last().ffill()
    
    # Weekly PnL is the difference from one week's equity to the next
    weekly_pnl = weekly_equity.diff().fillna(0)
    
    # Weekly drawdown = how far we are below the running max (in %)
    running_max_weekly = weekly_equity.cummax()
    weekly_drawdown_series = (weekly_equity - running_max_weekly) / running_max_weekly * 100
    
    # -----------------------------------------------------------------------------
    # 10. Create a DataFrame for Weekly Stats and Log Them
    # -----------------------------------------------------------------------------
    weekly_stats = pd.DataFrame({
        "WeeklyEquity": weekly_equity,
        "WeeklyPnL": weekly_pnl,
        "WeeklyDrawdown(%)": weekly_drawdown_series
    })
    
    # Print the first few rows in console
    print("\n=== Weekly Stats (head) ===")
    print(weekly_stats)
    
    # Optionally, save the entire weekly stats DataFrame to CSV
    weekly_stats.to_csv("weekly_stats.csv", index=True)
    print("\nSaved weekly stats to 'weekly_stats.csv'")

    # -----------------------------------------------------------------------------
    # 11. Plot and Save Charts to PNG
    # -----------------------------------------------------------------------------
    # Weekly PnL Chart
    plt.figure()
    plt.plot(weekly_pnl)
    plt.title("Weekly PnL")
    plt.xlabel("Date")
    plt.ylabel("PnL")
    plt.savefig("weekly_pnl_chart.png")
    plt.close()
    
    # Weekly Drawdown Chart
    plt.figure()
    plt.plot(weekly_drawdown_series)
    plt.title("Weekly Drawdown (%)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.savefig("weekly_drawdown_chart.png")
    plt.close()

    # Calculate max weekly drawdown (which is the min in the drawdown series, as it's negative)
    max_dd_weekly = weekly_drawdown_series.min()  # e.g. -15 means -15%
    
    # -----------------------------------------------------------------------------
    # 12. Print Summary Results
    # -----------------------------------------------------------------------------
    print("\n=== RESULTS ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Trade Multiplier: x{trade_multiplier}")
    print(f"Number of Trades: {num_trades}")
    print(f"Total Days: {total_days}")
    print(f"Estimated Trades per Year: {trades_per_year:.2f}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    if not cumulative_pnl.empty:
        print(f"Final Equity: ${cumulative_pnl.iloc[-1]:,.2f}")
    print(f"Average Return per Trade: {returns.mean()*100:.2f}%")
    print(f"Standard Deviation of Returns: {returns.std(ddof=1)*100:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown (Trade-based): {max_drawdown_val:.2f}%")
    print(f"Annualized Return (Geometric): {annual_return_pct:.2f}% (if not NaN)")
    
    print(f"\n*** Max Weekly Drawdown: {max_dd_weekly:.2f}% (negative means below running max)")

    # -----------------------------------------------------------------------------
    # 13. Optional: Trade Statistics
    # -----------------------------------------------------------------------------
    print("\nTrade Statistics (USD):")
    print(df['Profit'].describe())
    
    print("\nTrade Statistics (Returns %):")
    print((returns * 100).describe())


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run once with default parameters; will log and produce PNGs for weekly PnL & drawdown
    print(">> RUN 1: Default parameters (initial_capital=40K, multiplier=1x)\n")
    run_backtest("trades.csv", initial_capital=40000, trade_multiplier=1.0)

    # Run again with 1M capital, 25x multiplier; also logs weekly stats & produces PNGs
    print("\n\n>> RUN 2: 1M capital, 25x multiplier\n")
    run_backtest("trades.csv", initial_capital=1_000_000, trade_multiplier=25.0)
