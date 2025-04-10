import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import norm
import empyrical as ep  # For financial metrics

class EnhancedBacktester:
    def __init__(self, data, initial_capital=100000.0, commission=0.001, slippage=0.0005):
        """
        Initialize the backtester with historical data and parameters.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data and signals
            initial_capital (float): Starting capital
            commission (float): Commission rate as a percentage of transaction value
            slippage (float): Slippage as a percentage of transaction price
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.positions = pd.DataFrame(index=data.index)
        self.portfolio = pd.DataFrame(index=data.index)
        self.risk_metrics = {}
        self.trade_history = []
        
    def run_backtest(self, signal_column='Signal', price_column='Close', 
                     volatility_column='Volatility', position_sizing='volatility',
                     stop_loss=0.03, take_profit=0.05, max_position=0.2):
        """
        Run the backtest with the given parameters.
        
        Args:
            signal_column (str): Column containing trade signals (1=long, -1=short, 0=neutral)
            price_column (str): Column containing price data
            volatility_column (str): Column containing volatility estimates
            position_sizing (str): Method for position sizing ('equal', 'volatility', 'kelly')
            stop_loss (float): Stop loss as a percentage of entry price
            take_profit (float): Take profit as a percentage of entry price
            max_position (float): Maximum position size as a percentage of portfolio
        """
        # Ensure data is sorted by date
        self.data = self.data.sort_index()
        
        # Initialize portfolio and positions
        self.portfolio['Cash'] = self.initial_capital
        self.portfolio['Holdings'] = 0.0
        self.portfolio['Total'] = self.initial_capital
        self.positions['Signal'] = self.data[signal_column]
        self.positions['Price'] = self.data[price_column]
        self.positions['Position'] = 0.0
        
        current_position = 0
        entry_price = 0
        
        # Implement regime detection
        self.data['Regime'] = self._detect_market_regime()
        
        # Track open positions for stop-loss/take-profit
        open_positions = []
        
        for i in range(1, len(self.data)):
            # Previous day's portfolio value
            prev_cash = self.portfolio['Cash'].iloc[i-1]
            prev_holdings = self.portfolio['Holdings'].iloc[i-1]
            current_price = self.positions['Price'].iloc[i]
            
            # Update holdings value with current price
            self.portfolio['Holdings'].iloc[i] = current_position * current_price
            
            # Check stop-loss and take-profit for open positions
            self._check_stop_loss_take_profit(i, current_price, stop_loss, take_profit)
            
            # Get new signal
            signal = self.positions['Signal'].iloc[i]
            
            # Position sizing based on selected method
            if signal != 0 and signal != current_position:  # Only trade on new signals
                # Close existing position if any
                if current_position != 0:
                    # Calculate transaction cost with slippage
                    exit_price = current_price * (1 - current_position * self.slippage)
                    transaction_value = abs(current_position) * exit_price
                    commission_cost = transaction_value * self.commission
                    
                    # Update cash balance
                    self.portfolio['Cash'].iloc[i] = prev_cash + (current_position * exit_price) - commission_cost
                    
                    # Record the trade
                    self.trade_history.append({
                        'Entry Date': entry_date,
                        'Exit Date': self.data.index[i],
                        'Direction': 'Long' if current_position > 0 else 'Short',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Profit/Loss': (exit_price - entry_price) * current_position - commission_cost,
                        'Position Size': abs(current_position)
                    })
                    
                    # Reset current position
                    current_position = 0
                    
                # Open new position based on signal
                if signal != 0:
                    # Calculate position size
                    position_size = self._calculate_position_size(
                        i, position_sizing, volatility_column, max_position
                    )
                    
                    # Apply signal direction
                    position_size *= signal
                    
                    # Calculate entry price with slippage
                    entry_price = current_price * (1 + signal * self.slippage)
                    
                    # Calculate transaction cost
                    transaction_value = abs(position_size) * entry_price
                    commission_cost = transaction_value * self.commission
                    
                    # Update cash balance
                    self.portfolio['Cash'].iloc[i] = prev_cash - (position_size * entry_price) - commission_cost
                    
                    # Update current position
                    current_position = position_size
                    
                    # Record entry date for this trade
                    entry_date = self.data.index[i]
                else:
                    # No new position, maintain cash
                    self.portfolio['Cash'].iloc[i] = prev_cash
            else:
                # No change in position
                self.portfolio['Cash'].iloc[i] = prev_cash
            
            # Update positions
            self.positions['Position'].iloc[i] = current_position
            
            # Calculate total portfolio value
            self.portfolio['Total'].iloc[i] = self.portfolio['Cash'].iloc[i] + self.portfolio['Holdings'].iloc[i]
        
        # Calculate portfolio returns
        self.portfolio['Returns'] = self.portfolio['Total'].pct_change()
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        return self.portfolio, self.positions, self.risk_metrics
    
    def _calculate_position_size(self, index, method, volatility_column, max_position):
        """Calculate position size based on the specified method."""
        portfolio_value = self.portfolio['Total'].iloc[index-1]
        
        if method == 'equal':
            # Equal position sizing (percentage of portfolio)
            return max_position
        
        elif method == 'volatility':
            # Volatility-based position sizing (target volatility)
            target_volatility = 0.01  # Target 1% daily portfolio volatility
            asset_volatility = self.data[volatility_column].iloc[index]
            if asset_volatility > 0:
                position_size = target_volatility / asset_volatility
                return min(position_size, max_position)
            else:
                return max_position
        
        elif method == 'kelly':
            # Kelly criterion for position sizing
            # Calculate historical win rate and avg win/loss ratio
            # Using simplification for demonstration
            win_rate = 0.55  # Placeholder - should be calculated from signal performance
            avg_win_loss_ratio = 1.5  # Placeholder - should be calculated from signal performance
            kelly_fraction = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
            # Use half-Kelly for more conservative sizing
            return min(kelly_fraction * 0.5, max_position)
        
        else:
            # Default to equal sizing
            return max_position
    
    def _detect_market_regime(self):
        """
        Detect market regime (trending or mean-reverting).
        Returns a Series with regime labels.
        """
        # Using Hurst exponent as a simple regime detection method
        # H > 0.5 indicates trending, H < 0.5 indicates mean-reverting
        
        # Calculate returns
        returns = self.data['Close'].pct_change().dropna()
        
        # Simple rolling autocorrelation as proxy for regime
        # Positive autocorrelation suggests trending, negative suggests mean-reverting
        window = 20
        rolling_autocorr = returns.rolling(window=window+1).apply(
            lambda x: pd.Series(x).autocorr(), raw=True
        )
        
        # Assign regimes based on autocorrelation
        regimes = pd.Series(index=self.data.index)
        regimes[rolling_autocorr > 0.2] = 'Trending'
        regimes[rolling_autocorr < -0.2] = 'Mean-Reverting'
        regimes.fillna('Unknown', inplace=True)
        
        return regimes
    
    def _check_stop_loss_take_profit(self, index, current_price, stop_loss, take_profit):
        """Check if any open positions hit stop-loss or take-profit levels."""
        position = self.positions['Position'].iloc[index-1]
        
        if position == 0:
            return
        
        entry_price = None
        # Find the most recent entry price for the current position
        for i in range(index-1, 0, -1):
            if self.positions['Position'].iloc[i] != self.positions['Position'].iloc[i-1]:
                entry_price = self.positions['Price'].iloc[i]
                break
        
        if entry_price is None:
            return
        
        # Check stop loss
        if position > 0:  # Long position
            stop_price = entry_price * (1 - stop_loss)
            take_price = entry_price * (1 + take_profit)
            
            if current_price <= stop_price:
                # Trigger stop loss
                self.positions['Signal'].iloc[index] = 0  # Signal to close position
                print(f"Stop loss triggered at {self.data.index[index]}: Price {current_price:.2f} <= Stop {stop_price:.2f}")
            
            elif current_price >= take_price:
                # Trigger take profit
                self.positions['Signal'].iloc[index] = 0  # Signal to close position
                print(f"Take profit triggered at {self.data.index[index]}: Price {current_price:.2f} >= Take {take_price:.2f}")
        
        elif position < 0:  # Short position
            stop_price = entry_price * (1 + stop_loss)
            take_price = entry_price * (1 - take_profit)
            
            if current_price >= stop_price:
                # Trigger stop loss
                self.positions['Signal'].iloc[index] = 0  # Signal to close position
                print(f"Stop loss triggered at {self.data.index[index]}: Price {current_price:.2f} >= Stop {stop_price:.2f}")
            
            elif current_price <= take_price:
                # Trigger take profit
                self.positions['Signal'].iloc[index] = 0  # Signal to close position
                print(f"Take profit triggered at {self.data.index[index]}: Price {current_price:.2f} <= Take {take_price:.2f}")
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        returns = self.portfolio['Returns'].dropna()
        
        # Basic metrics
        self.risk_metrics['Total Return (%)'] = ((self.portfolio['Total'].iloc[-1] / self.initial_capital) - 1) * 100
        self.risk_metrics['CAGR (%)'] = (((self.portfolio['Total'].iloc[-1] / self.initial_capital) ** 
                                         (252 / len(self.portfolio))) - 1) * 100
        self.risk_metrics['Annualized Volatility (%)'] = returns.std() * np.sqrt(252) * 100
        self.risk_metrics['Sharpe Ratio'] = ep.sharpe_ratio(returns, annualization=252)
        self.risk_metrics['Sortino Ratio'] = ep.sortino_ratio(returns, annualization=252)
        self.risk_metrics['Max Drawdown (%)'] = ep.max_drawdown(returns) * 100
        self.risk_metrics['Calmar Ratio'] = ep.calmar_ratio(returns, annualization=252)
        
        # Calculate win rate from trade history
        if self.trade_history:
            profitable_trades = sum(1 for trade in self.trade_history if trade['Profit/Loss'] > 0)
            total_trades = len(self.trade_history)
            self.risk_metrics['Win Rate (%)'] = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            self.risk_metrics['Total Trades'] = total_trades
            
            # Calculate average profit/loss
            avg_profit = np.mean([t['Profit/Loss'] for t in self.trade_history if t['Profit/Loss'] > 0]) if profitable_trades > 0 else 0
            avg_loss = np.mean([t['Profit/Loss'] for t in self.trade_history if t['Profit/Loss'] <= 0]) if total_trades - profitable_trades > 0 else 0
            
            self.risk_metrics['Avg Profit'] = avg_profit
            self.risk_metrics['Avg Loss'] = avg_loss
            if avg_loss != 0:
                self.risk_metrics['Profit/Loss Ratio'] = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            # Calculate average holding period
            holding_periods = [(trade['Exit Date'] - trade['Entry Date']).days for trade in self.trade_history]
            self.risk_metrics['Avg Holding Period (Days)'] = np.mean(holding_periods) if holding_periods else 0
        
        # Calculate beta to market if market data is available
        if 'Market_Return' in self.data.columns:
            market_returns = self.data['Market_Return'].loc[returns.index]
            self.risk_metrics['Beta'] = returns.cov(market_returns) / market_returns.var()
            self.risk_metrics['Alpha (%)'] = (self.risk_metrics['CAGR (%)'] - 
                                            self.risk_metrics['Beta'] * market_returns.mean() * 252 * 100)
    
    def plot_equity_curve(self):
        """Plot the equity curve with drawdowns."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.portfolio['Total'], label='Portfolio Value', color='blue')
        
        # Plot drawdowns
        drawdown = (self.portfolio['Total'] / self.portfolio['Total'].cummax() - 1)
        ax.fill_between(self.portfolio.index, 0, drawdown * self.portfolio['Total'].max(), 
                        alpha=0.3, color='red', label='Drawdowns')
        
        ax.set_title('Portfolio Performance')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_returns_distribution(self):
        """Plot the distribution of returns vs normal distribution."""
        returns = self.portfolio['Returns'].dropna()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram
        returns.hist(ax=ax, bins=50, alpha=0.5, density=True, label='Returns Distribution')
        
        # Fit normal distribution and plot
        mu, std = norm.fit(returns.dropna())
        x = np.linspace(returns.min(), returns.max(), 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, label=f'Normal: $\mu={mu:.4f}$, $\sigma={std:.4f}$')
        
        ax.set_title('Returns Distribution vs. Normal Distribution')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def plot_regime_performance(self):
        """Plot performance split by market regime."""
        self.portfolio['Regime'] = self.data['Regime']
        
        # Calculate returns by regime
        regime_returns = {}
        for regime in self.portfolio['Regime'].unique():
            if pd.notna(regime):
                mask = self.portfolio['Regime'] == regime
                regime_returns[regime] = self.portfolio.loc[mask, 'Returns'].dropna()
        
        # Create subplots for each regime
        fig, axes = plt.subplots(len(regime_returns), 1, figsize=(12, 4*len(regime_returns)))
        if len(regime_returns) == 1:
            axes = [axes]  # Make it iterable if only one regime
        
        for (regime, ret), ax in zip(regime_returns.items(), axes):
            # Calculate cumulative returns
            cumulative = (1 + ret).cumprod()
            
            # Plot cumulative returns
            ax.plot(cumulative.index, cumulative, label=f'{regime} Cumulative Return')
            
            # Calculate and display key metrics
            total_return = (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0
            vol = ret.std() * np.sqrt(252) * 100 if len(ret) > 0 else 0
            sharpe = ret.mean() / ret.std() * np.sqrt(252) if len(ret) > 0 and ret.std() > 0 else 0
            
            ax.set_title(f'{regime} Performance: Return={total_return:.2f}%, Vol={vol:.2f}%, Sharpe={sharpe:.2f}')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def run_monte_carlo_simulation(self, n_simulations=1000, time_horizon=252):
        """
        Run Monte Carlo simulation to estimate portfolio value distribution.
        
        Args:
            n_simulations (int): Number of simulations
            time_horizon (int): Number of days to simulate
            
        Returns:
            DataFrame with simulation results
        """
        returns = self.portfolio['Returns'].dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random returns
        np.random.seed(42)
        sim_returns = np.random.normal(mu, sigma, (time_horizon, n_simulations))
        
        # Calculate cumulative returns
        current_value = self.portfolio['Total'].iloc[-1]
        sim_paths = np.zeros((time_horizon + 1, n_simulations))
        sim_paths[0] = current_value
        
        for i in range(time_horizon):
            sim_paths[i+1] = sim_paths[i] * (1 + sim_returns[i])
        
        # Convert to DataFrame
        sim_df = pd.DataFrame(sim_paths, columns=[f'Sim_{i}' for i in range(n_simulations)])
        
        # Add dates
        last_date = self.portfolio.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(time_horizon + 1)]
        
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.DatetimeIndex(future_dates)
        
        sim_df.index = future_dates
        
        # Calculate statistics
        final_values = sim_df.iloc[-1]
        self.risk_metrics['MC_Median'] = final_values.median()
        self.risk_metrics['MC_5th_Percentile'] = final_values.quantile(0.05)
        self.risk_metrics['MC_95th_Percentile'] = final_values.quantile(0.95)
        self.risk_metrics['MC_Expected_Value'] = final_values.mean()
        
        return sim_df
    
    def plot_monte_carlo(self, sim_df):
        """Plot Monte Carlo simulation results."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot all simulations with low alpha
        for i in range(sim_df.shape[1]):
            ax.plot(sim_df.index, sim_df.iloc[:, i], linewidth=0.5, alpha=0.05, color='blue')
        
        # Plot historical data
        historical_dates = self.portfolio.index
        ax.plot(historical_dates, self.portfolio['Total'], color='black', linewidth=2, label='Historical')
        
        # Plot median, 5th and 95th percentiles
        ax.plot(sim_df.index, sim_df.median(axis=1), color='red', linewidth=2, label='Median')
        ax.plot(sim_df.index, sim_df.quantile(0.05, axis=1), color='green', linewidth=2, linestyle='--', label='5th Percentile')
        ax.plot(sim_df.index, sim_df.quantile(0.95, axis=1), color='purple', linewidth=2, linestyle='--', label='95th Percentile')
        
        ax.set_title('Monte Carlo Simulation of Portfolio Value')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def stress_test(self, scenario_name, scenario_returns):
        """
        Perform stress test using historical scenario returns.
        
        Args:
            scenario_name (str): Name of the scenario (e.g., '2008 Financial Crisis')
            scenario_returns (pd.Series): Series of market returns during the scenario
            
        Returns:
            DataFrame with portfolio performance during the scenario
        """
        # Create a copy of the stress test returns
        scenario_df = scenario_returns.to_frame('Market')
        
        # Estimate portfolio returns using beta
        if 'Beta' not in self.risk_metrics:
            self.risk_metrics['Beta'] = 1.0  # Default to market beta if not calculated
            
        scenario_df['Portfolio'] = scenario_df['Market'] * self.risk_metrics['Beta']
        
        # Calculate drawdowns
        scenario_df['Market_Cumulative'] = (1 + scenario_df['Market']).cumprod()
        scenario_df['Portfolio_Cumulative'] = (1 + scenario_df['Portfolio']).cumprod()
        
        scenario_df['Market_Drawdown'] = scenario_df['Market_Cumulative'] / scenario_df['Market_Cumulative'].cummax() - 1
        scenario_df['Portfolio_Drawdown'] = scenario_df['Portfolio_Cumulative'] / scenario_df['Portfolio_Cumulative'].cummax() - 1
        
        # Calculate stress test metrics
        self.risk_metrics[f'{scenario_name}_Market_Drawdown'] = scenario_df['Market_Drawdown'].min() * 100
        self.risk_metrics[f'{scenario_name}_Portfolio_Drawdown'] = scenario_df['Portfolio_Drawdown'].min() * 100
        self.risk_metrics[f'{scenario_name}_Portfolio_Return'] = (scenario_df['Portfolio_Cumulative'].iloc[-1] - 1) * 100
        
        return scenario_df
    
    def plot_stress_test(self, scenario_df, scenario_name):
        """Plot stress test results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot cumulative returns
        ax1.plot(scenario_df.index, scenario_df['Market_Cumulative'], label='Market', color='blue')
        ax1.plot(scenario_df.index, scenario_df['Portfolio_Cumulative'], label='Portfolio', color='green')
        ax1.set_title(f'Cumulative Returns during {scenario_name}')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        
        # Plot drawdowns
        ax2.fill_between(scenario_df.index, 0, scenario_df['Market_Drawdown'], label='Market Drawdown', alpha=0.3, color='blue')
        ax2.fill_between(scenario_df.index, 0, scenario_df['Portfolio_Drawdown'], label='Portfolio Drawdown', alpha=0.3, color='green')
        ax2.set_title(f'Drawdowns during {scenario_name}')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(min(scenario_df['Market_Drawdown'].min(), scenario_df['Portfolio_Drawdown'].min()) * 1.1, 0.05)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def report_performance(self):
        """Print a detailed performance report."""
        print("=" * 50)
        print("PERFORMANCE REPORT")
        print("=" * 50)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${self.portfolio['Total'].iloc[-1]:,.2f}")
        print(f"Total Return: {self.risk_metrics['Total Return (%)']:.2f}%")
        print(f"CAGR: {self.risk_metrics['CAGR (%)']:.2f}%")
        print(f"Volatility (Annualized): {self.risk_metrics['Annualized Volatility (%)']:.2f}%")
        print(f"Sharpe Ratio: {self.risk_metrics['Sharpe Ratio']:.2f}")
        print(f"Sortino Ratio: {self.risk_metrics['Sortino Ratio']:.2f}")
        print(f"Max Drawdown: {self.risk_metrics['Max Drawdown (%)']:.2f}%")
        print(f"Calmar Ratio: {self.risk_metrics['Calmar Ratio']:.2f}")
        
        if 'Win Rate (%)' in self.risk_metrics:
            print("\nTRADE STATISTICS")
            print("-" * 50)
            print(f"Total Trades: {self.risk_metrics['Total Trades']}")
            print(f"Win Rate: {self.risk_metrics['Win Rate (%)']:.2f}%")
            print(f"Average Profit: ${self.risk_metrics['Avg Profit']:,.2f}")
            print(f"Average Loss: ${self.risk_metrics['Avg Loss']:,.2f}")
            print(f"Profit/Loss Ratio: {self.risk_metrics['Profit/Loss Ratio']:.2f}")
            print(f"Average Holding Period: {self.risk_metrics['Avg Holding Period (Days)']:.1f} days")
        
        if 'MC_Median' in self.risk_metrics:
            print("\nMONTE CARLO SIMULATION RESULTS")
            print("-" * 50)
            print(f"Expected Portfolio Value: ${self.risk_metrics['MC_Expected_Value']:,.2f}")
            print(f"Median Portfolio Value: ${self.risk_metrics['MC_Median']:,.2f}")
            print(f"5th Percentile: ${self.risk_metrics['MC_5th_Percentile']:,.2f}")
            print(f"95th Percentile: ${self.risk_metrics['MC_95th_Percentile']:,.2f}")
        
        print("=" * 50)


class AdvancedTradingStrategies:
    """Class for implementing advanced trading strategies."""
    
    @staticmethod
    def trend_following(data, fast_period=20, slow_period=50, signal_threshold=0):
        """
        Implement a trend following strategy based on moving average crossovers.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            fast_period (int): Fast moving average period
            slow_period (int): Slow moving average period
            signal_threshold (float): Threshold for generating signals
            
        Returns:
            pd.Series: Series of trade signals (-1, 0, 1)
        """
        df = data.copy()
        df['MA_Fast'] = df['Close'].rolling(window=fast_period).mean()
        df['MA_Slow'] = df['Close'].rolling(window=slow_period).mean()
        
        # Calculate moving average crossover
        df['MA_Diff'] = df['MA_Fast'] - df['MA_Slow']
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[df['MA_Diff'] > signal_threshold] = 1  # Long signal
        signals[df['MA_Diff'] < -signal_threshold] = -1  # Short signal
        
        return signals
    
    @staticmethod
    def mean_reversion(data, lookback=20, entry_zscore=2.0, exit_zscore=0.5):
        """
        Implement a mean reversion strategy based on z-scores.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            lookback (int): Lookback period for calculating mean and std
            entry_zscore (float): Z-score threshold for entry
            exit_zscore (float): Z-score threshold for exit
            
        Returns:
            pd.Series: Series of trade signals (-1, 0, 1)
        """
        df = data.copy()
        
        # Calculate rolling mean and standard deviation
        df['Rolling_Mean'] = df['Close'].rolling(window=lookback).mean()
        df['Rolling_Std'] = df['Close'].rolling(window=lookback).std()
        
        # Calculate z-score
        df['ZScore'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Short when price is too high (positive z-score)
        signals[df['ZScore'] > entry_zscore] = -1
        
        # Long when price is too low (negative z-score)
        signals[df['ZScore'] < -entry_zscore] = 1
        
        # Exit signals
        for i in range(1, len(signals)):
            if signals.iloc[i-1] == -1 and df['ZScore'].iloc[i] < exit_zscore:
                signals.iloc[i] = 0  # Exit short
            elif signals.iloc[i-1] == 1 and df['ZScore'].iloc[i] > -exit_zscore:
                signals.iloc[i] = 0  # Exit long
        
        return signals
    @staticmethod
    def volatility_breakout(data, lookback=20, multiplier=2.0):
        """
        Implement a volatility breakout strategy.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            lookback (int): Lookback period for calculating volatility
            multiplier (float): Volatility multiplier for determining breakout
            
        Returns:
            pd.Series: Series of trade signals (-1, 0, 1)
        """
        df = data.copy()
        
        # Calculate Average True Range (ATR) as volatility measure
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=lookback).mean()
        
        # Calculate upper and lower bands
        df['Upper'] = df['Close'].shift(1) + (multiplier * df['ATR'])
        df['Lower'] = df['Close'].shift(1) - (multiplier * df['ATR'])
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Long when price breaks above upper band
        signals[(df['Close'] > df['Upper']) & (df['Close'].shift(1) <= df['Upper'].shift(1))] = 1
        
        # Short when price breaks below lower band
        signals[(df['Close'] < df['Lower']) & (df['Close'].shift(1) >= df['Lower'].shift(1))] = -1
        
        # Exit signals
        for i in range(1, len(signals)):
            # Exit long if price closes below middle band
            if signals.iloc[i-1] == 1 and df['Close'].iloc[i] < (df['Upper'].iloc[i] + df['Lower'].iloc[i]) / 2:
                signals.iloc[i] = 0
            # Exit short if price closes above middle band
            elif signals.iloc[i-1] == -1 and df['Close'].iloc[i] > (df['Upper'].iloc[i] + df['Lower'].iloc[i]) / 2:
                signals.iloc[i] = 0
        
        return signals
    
    @staticmethod
    def momentum_strategy(data, momentum_period=20, signal_threshold=0):
        """
        Implement a momentum strategy based on rate of change.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            momentum_period (int): Lookback period for momentum calculation
            signal_threshold (float): Threshold for generating signals
            
        Returns:
            pd.Series: Series of trade signals (-1, 0, 1)
        """
        df = data.copy()
        
        # Calculate rate of change as momentum indicator
        df['Momentum'] = df['Close'].pct_change(periods=momentum_period)
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[df['Momentum'] > signal_threshold] = 1  # Long when momentum is positive
        signals[df['Momentum'] < -signal_threshold] = -1  # Short when momentum is negative
        
        return signals
    
    @staticmethod
    def dual_timeframe_rsi(data, rsi_period=14, fast_ma=5, slow_ma=20, overbought=70, oversold=30):
        """
        Implement a dual timeframe RSI strategy.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            rsi_period (int): Period for RSI calculation
            fast_ma (int): Fast moving average period for RSI
            slow_ma (int): Slow moving average period for RSI
            overbought (float): Overbought threshold for RSI
            oversold (float): Oversold threshold for RSI
            
        Returns:
            pd.Series: Series of trade signals (-1, 0, 1)
        """
        df = data.copy()
        
        # Calculate daily price changes
        delta = df['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate RSI moving averages
        df['RSI_Fast'] = df['RSI'].rolling(window=fast_ma).mean()
        df['RSI_Slow'] = df['RSI'].rolling(window=slow_ma).mean()
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Long signal: RSI crosses above oversold and fast MA crosses above slow MA
        signals[(df['RSI'] > oversold) & 
                (df['RSI'].shift(1) <= oversold) & 
                (df['RSI_Fast'] > df['RSI_Slow'])] = 1
        
        # Short signal: RSI crosses below overbought and fast MA crosses below slow MA
        signals[(df['RSI'] < overbought) & 
                (df['RSI'].shift(1) >= overbought) & 
                (df['RSI_Fast'] < df['RSI_Slow'])] = -1
        
        return signals
    
    @staticmethod
    def volume_price_trend(data, vpt_period=14, signal_ma=9):
        """
        Implement a volume-price trend strategy.
        
        Args:
            data (pd.DataFrame): DataFrame with price and volume data
            vpt_period (int): Period for VPT calculation
            signal_ma (int): Moving average period for signal line
            
        Returns:
            pd.Series: Series of trade signals (-1, 0, 1)
        """
        df = data.copy()
        
        # Calculate price percent change
        df['PriceChg'] = df['Close'].pct_change()
        
        # Calculate VPT
        df['VPT'] = df['Volume'] * df['PriceChg']
        df['CumVPT'] = df['VPT'].cumsum()
        
        # Calculate VPT moving averages
        df['VPT_MA'] = df['CumVPT'].rolling(window=vpt_period).mean()
        df['Signal'] = df['VPT_MA'].rolling(window=signal_ma).mean()
        
        # Generate signals based on VPT and signal line crossovers
        signals = pd.Series(0, index=df.index)
        signals[(df['VPT_MA'] > df['Signal']) & (df['VPT_MA'].shift(1) <= df['Signal'].shift(1))] = 1
        signals[(df['VPT_MA'] < df['Signal']) & (df['VPT_MA'].shift(1) >= df['Signal'].shift(1))] = -1
        
        return signals
    
    @staticmethod
    def adaptive_strategy(data, short_period=10, long_period=40, volatility_window=20, 
                          trending_threshold=0.2, reverting_threshold=-0.2):
        """
        Implement a strategy that adapts to market conditions.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            short_period (int): Short moving average period
            long_period (int): Long moving average period
            volatility_window (int): Window for calculating volatility
            trending_threshold (float): Autocorrelation threshold for trending markets
            reverting_threshold (float): Autocorrelation threshold for mean-reverting markets
            
        Returns:
            pd.Series: Series of trade signals (-1, 0, 1)
        """
        df = data.copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Detect market regime using rolling autocorrelation
        df['Autocorr'] = df['Returns'].rolling(volatility_window+1).apply(
            lambda x: pd.Series(x).autocorr(), raw=True
        )
        
        # Calculate moving averages for trend following
        df['MA_Short'] = df['Close'].rolling(window=short_period).mean()
        df['MA_Long'] = df['Close'].rolling(window=long_period).mean()
        
        # Calculate z-score for mean reversion
        df['Mean'] = df['Close'].rolling(window=volatility_window).mean()
        df['Std'] = df['Close'].rolling(window=volatility_window).std()
        df['ZScore'] = (df['Close'] - df['Mean']) / df['Std']
        
        # Generate signals based on detected regime
        signals = pd.Series(0, index=df.index)
        
        # Trending market: use moving average crossover
        trending_mask = df['Autocorr'] > trending_threshold
        signals[trending_mask & (df['MA_Short'] > df['MA_Long'])] = 1
        signals[trending_mask & (df['MA_Short'] < df['MA_Long'])] = -1
        
        # Mean-reverting market: use zscore
        reverting_mask = df['Autocorr'] < reverting_threshold
        signals[reverting_mask & (df['ZScore'] < -2)] = 1  # Buy when significantly below mean
        signals[reverting_mask & (df['ZScore'] > 2)] = -1  # Sell when significantly above mean
        
        return signals
    
    @staticmethod
    def ensemble_strategy(data, strategies_list, weights=None):
        """
        Implement an ensemble strategy that combines multiple strategies.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            strategies_list (list): List of strategy functions to combine
            weights (list): List of weights for each strategy (equal if None)
            
        Returns:
            pd.Series: Series of trade signals (-1, 0, 1)
        """
        if weights is None:
            weights = [1.0 / len(strategies_list)] * len(strategies_list)
            
        if len(weights) != len(strategies_list):
            raise ValueError("Number of weights must match number of strategies")
            
        # Get signals from all strategies
        all_signals = []
        for strategy_func in strategies_list:
            signals = strategy_func(data)
            all_signals.append(signals)
            
        # Create a DataFrame of all signals
        signals_df = pd.DataFrame(all_signals).T
        
        # Calculate weighted average signal
        weighted_signal = pd.Series(0.0, index=data.index)
        for i, weight in enumerate(weights):
            weighted_signal += signals_df.iloc[:, i] * weight
            
        # Convert to discrete signals
        final_signals = pd.Series(0, index=data.index)
        final_signals[weighted_signal > 0.3] = 1  # Long if weighted signal is sufficiently positive
        final_signals[weighted_signal < -0.3] = -1  # Short if weighted signal is sufficiently negative
        
        return final_signals
       
