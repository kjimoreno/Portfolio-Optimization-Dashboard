import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import ta
from datetime import datetime
import time
from requests.exceptions import HTTPError

class DataProcessor:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def fetch_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data for given symbols with rate limit handling
        """
        data = pd.DataFrame()
        valid_symbols = []
        error_messages = []
        
        # Validate dates
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if end > pd.Timestamp.now():
                end = pd.Timestamp.now()
            if start > end:
                start = end - pd.Timedelta(days=365)
            print(f"Using date range: {start.date()} to {end.date()}")
        except Exception as e:
            error_msg = f"Invalid date format: {str(e)}"
            print(error_msg)
            error_messages.append(error_msg)
            return pd.DataFrame()
        
        print(f"\nFetching data for symbols: {symbols}")
        print(f"Date range: {start_date} to {end_date}")
        
        for symbol in symbols:
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Remove any special characters and convert to uppercase
                    symbol = symbol.strip().upper()
                    if not symbol:
                        continue
                        
                    # Remove any '$' prefix if present
                    symbol = symbol.replace('$', '')
                    
                    print(f"\nAttempting to fetch data for {symbol} (attempt {attempt + 1}/{max_retries})")
                    
                    # Add delay between requests to avoid rate limits
                    if attempt > 0:
                        print(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                    
                    # Try to get ticker info first to validate symbol
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if not info:
                        error_msg = f"Invalid symbol: {symbol}"
                        print(error_msg)
                        error_messages.append(error_msg)
                        break
                    
                    print(f"Found valid ticker for {symbol}")
                    
                    # Fetch historical data
                    df = ticker.history(start=start_date, end=end_date)
                    
                    if df.empty:
                        error_msg = f"No data found for {symbol} between {start_date} and {end_date}"
                        print(error_msg)
                        error_messages.append(error_msg)
                        break
                    
                    print(f"Successfully fetched {len(df)} rows of data for {symbol}")
                    print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
                    
                    data[symbol] = df['Close']
                    valid_symbols.append(symbol)
                    break  # Success, exit retry loop
                    
                except HTTPError as e:
                    if e.response.status_code == 429:  # Too Many Requests
                        if attempt < max_retries - 1:
                            print(f"Rate limit hit, waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            error_msg = f"Rate limit hit for {symbol} after {max_retries} attempts"
                            print(error_msg)
                            error_messages.append(error_msg)
                    else:
                        error_msg = f"HTTP error fetching data for {symbol}: {str(e)}"
                        print(error_msg)
                        error_messages.append(error_msg)
                        break
                except Exception as e:
                    error_msg = f"Error fetching data for {symbol}: {str(e)}"
                    print(error_msg)
                    error_messages.append(error_msg)
                    break
        
        if data.empty:
            print("\nError Summary:")
            print("\n".join(error_messages))
            print("\nNo valid data was fetched for any symbols")
            return pd.DataFrame()
            
        # Ensure all dates are aligned
        data = data.dropna()
        
        if data.empty:
            print("No overlapping data found for the selected symbols")
            return pd.DataFrame()
            
        print(f"\nFinal dataset shape: {data.shape}")
        print(f"Valid symbols: {valid_symbols}")
        print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
        return data
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns
        """
        if prices.empty:
            return pd.DataFrame()
        return prices.pct_change().dropna()
    
    def calculate_metrics(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate key portfolio metrics
        """
        if returns.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        # Calculate mean returns
        mean_returns = returns.mean()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        return mean_returns, cov_matrix, corr_matrix
    
    def add_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the price data
        """
        if prices.empty:
            return pd.DataFrame()
            
        indicators = pd.DataFrame(index=prices.index)
        
        for column in prices.columns:
            try:
                # RSI
                indicators[f'{column}_RSI'] = ta.momentum.RSIIndicator(prices[column]).rsi()
                
                # MACD
                macd = ta.trend.MACD(prices[column])
                indicators[f'{column}_MACD'] = macd.macd()
                indicators[f'{column}_MACD_Signal'] = macd.macd_signal()
                
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(prices[column])
                indicators[f'{column}_BB_High'] = bollinger.bollinger_hband()
                indicators[f'{column}_BB_Low'] = bollinger.bollinger_lband()
                
                # Moving Averages
                indicators[f'{column}_SMA_20'] = ta.trend.SMAIndicator(prices[column], window=20).sma_indicator()
                indicators[f'{column}_SMA_50'] = ta.trend.SMAIndicator(prices[column], window=50).sma_indicator()
            except Exception as e:
                print(f"Error calculating indicators for {column}: {str(e)}")
        
        return indicators
    
    def calculate_var_cvar(self, returns: pd.DataFrame, confidence_level: float = 0.95, weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR
        """
        if returns.empty:
            return {'VaR': 0.0, 'CVaR': 0.0}
            
        if weights is None:
            # If no weights provided, use equal weights
            weights = {col: 1/len(returns.columns) for col in returns.columns}
        
        # Ensure all weights sum to 1
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Convert weights to numpy array in the same order as returns columns
        weights_array = np.array([weights[col] for col in returns.columns])
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights_array)
        
        # Remove any NaN values
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) == 0:
            return {'VaR': 0.0, 'CVaR': 0.0}
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Calculate CVaR
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return {
            'VaR': var,
            'CVaR': cvar
        }
    
    def calculate_sharpe_ratio(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """
        Calculate Sharpe ratio for a given portfolio
        """
        if returns.empty:
            return 0.0
            
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate annualized metrics
        annualized_return = portfolio_returns.mean() * 252
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        if annualized_volatility == 0:
            return 0.0
            
        # Calculate Sharpe ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        return sharpe_ratio
    
    def calculate_risk_contribution(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk contribution for each asset in the portfolio
        """
        if returns.empty:
            return {col: 0.0 for col in returns.columns}
            
        # Convert weights to numpy array
        weights_array = np.array(list(weights.values()))
        
        # Ensure weights sum to 1
        weights_array = weights_array / np.sum(weights_array)
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(returns.cov() * 252, weights_array)))
        
        if portfolio_volatility == 0:
            return {col: 0.0 for col in returns.columns}
            
        # Calculate marginal risk contribution
        marginal_risk_contribution = np.dot(returns.cov() * 252, weights_array) / portfolio_volatility
        
        # Calculate risk contribution
        risk_contribution = weights_array * marginal_risk_contribution
        
        # Convert to dictionary with asset names
        return dict(zip(returns.columns, risk_contribution)) 