import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, List, Dict
import plotly.graph_objects as go
from dataclasses import dataclass

@dataclass
class PortfolioMetrics:
    returns: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    weights: np.ndarray

class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio returns, volatility, and downside deviation
        """
        # Calculate portfolio returns
        portfolio_returns = self.returns.dot(weights)
        
        # Calculate annualized metrics
        portfolio_returns_annual = portfolio_returns.mean() * 252
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Calculate downside deviation
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        
        return portfolio_returns_annual, portfolio_volatility, downside_std
    
    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate negative Sharpe ratio (for minimization)
        """
        portfolio_returns, portfolio_volatility, _ = self.calculate_portfolio_metrics(weights)
        sharpe_ratio = (portfolio_returns - self.risk_free_rate) / portfolio_volatility
        return -sharpe_ratio
    
    def negative_sortino_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate negative Sortino ratio (for minimization)
        """
        portfolio_returns, _, downside_std = self.calculate_portfolio_metrics(weights)
        sortino_ratio = (portfolio_returns - self.risk_free_rate) / downside_std
        return -sortino_ratio
    
    def risk_parity_objective(self, weights: np.ndarray) -> float:
        """
        Calculate risk parity objective (minimize the difference in risk contribution)
        """
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        risk_contribution = weights * (np.dot(self.returns.cov() * 252, weights) / portfolio_volatility)
        risk_target = portfolio_volatility / self.n_assets
        return np.sum((risk_contribution - risk_target)**2)
    
    def optimize_portfolio(self, method: str = 'sharpe', max_position_size: float = 1.0) -> PortfolioMetrics:
        """
        Optimize portfolio weights based on specified method
        """
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        )
   
        bounds = tuple((0, max_position_size) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        if method == 'sharpe':
            result = minimize(
                self.negative_sharpe_ratio,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif method == 'min_var':
            result = minimize(
                lambda w: np.sqrt(np.dot(w.T, np.dot(self.returns.cov() * 252, w))),
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif method == 'sortino':
            result = minimize(
                self.negative_sortino_ratio,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif method == 'equal':
            result = type('Result', (), {'x': np.array([1/self.n_assets] * self.n_assets)})()
        elif method == 'risk_parity':
            result = minimize(
                self.risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimal_weights = result.x
        portfolio_returns, portfolio_volatility, downside_std = self.calculate_portfolio_metrics(optimal_weights)
        sharpe_ratio = (portfolio_returns - self.risk_free_rate) / portfolio_volatility
        sortino_ratio = (portfolio_returns - self.risk_free_rate) / downside_std
        
        return PortfolioMetrics(
            returns=portfolio_returns,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            weights=optimal_weights
        )
    
    def generate_efficient_frontier(self, n_points: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate points for the efficient frontier
        """
        returns_list = []
        volatilities_list = []
        
        # Generate random portfolios
        for _ in range(n_points):
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)
            
            portfolio_returns, portfolio_volatility, _ = self.calculate_portfolio_metrics(weights)
            returns_list.append(portfolio_returns)
            volatilities_list.append(portfolio_volatility)
        
        return returns_list, volatilities_list
    
    def plot_efficient_frontier(self) -> go.Figure:
        """
        Create an interactive efficient frontier plot
        """
        returns_list, volatilities_list = self.generate_efficient_frontier()
        
        # Get optimal portfolios
        sharpe_optimal = self.optimize_portfolio(method='sharpe')
        min_var_optimal = self.optimize_portfolio(method='min_var')
        sortino_optimal = self.optimize_portfolio(method='sortino')
        
        fig = go.Figure()
        
        # Add scatter plot of random portfolios
        fig.add_trace(go.Scatter(
            x=volatilities_list,
            y=returns_list,
            mode='markers',
            name='Random Portfolios',
            marker=dict(color='blue', size=8, opacity=0.5)
        ))
        
        # Add optimal portfolios
        fig.add_trace(go.Scatter(
            x=[sharpe_optimal.volatility],
            y=[sharpe_optimal.returns],
            mode='markers',
            name='Maximum Sharpe Portfolio',
            marker=dict(color='red', size=12)
        ))
        
        fig.add_trace(go.Scatter(
            x=[min_var_optimal.volatility],
            y=[min_var_optimal.returns],
            mode='markers',
            name='Minimum Variance Portfolio',
            marker=dict(color='green', size=12)
        ))
        
        fig.add_trace(go.Scatter(
            x=[sortino_optimal.volatility],
            y=[sortino_optimal.returns],
            mode='markers',
            name='Maximum Sortino Portfolio',
            marker=dict(color='purple', size=12)
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Portfolio Volatility',
            yaxis_title='Portfolio Returns',
            showlegend=True
        )
        
        return fig
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """
        Get optimal portfolio weights with asset names
        """
        optimal = self.optimize_portfolio(method='sharpe')
        return dict(zip(self.returns.columns, optimal.weights)) 