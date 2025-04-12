import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from data_processor import DataProcessor
from portfolio_optimizer import PortfolioOptimizer

# Set page config
st.set_page_config(
    page_title="Advanced Portfolio Optimization",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Advanced Portfolio Optimization")
st.markdown("""
    This application implements Modern Portfolio Theory and advanced quantitative finance concepts
    to help you optimize your investment portfolio.
""")

# Sidebar
st.sidebar.header("Portfolio Configuration")

# Asset selection with custom input
st.sidebar.subheader("Asset Selection")
default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']
selected_symbols = st.sidebar.multiselect(
    "Select from Popular Assets",
    options=default_symbols,
    default=default_symbols[:5]
)

# Custom stock input
custom_symbols = st.sidebar.text_input(
    "Add Custom Stocks (comma-separated)",
    help="Enter stock symbols separated by commas (e.g., 'AAPL,MSFT,GOOGL')"
)

# Combine selected and custom symbols
if custom_symbols:
    custom_symbols = [symbol.strip() for symbol in custom_symbols.split(',')]
    selected_symbols = list(set(selected_symbols + custom_symbols))

# Date range selection
st.sidebar.subheader("Time Period")
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Last year of data

# Initialize date range in session state if not exists
if 'date_range' not in st.session_state:
    st.session_state.date_range = (start_date, end_date)

# Get date range from user input
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=st.session_state.date_range,
    max_value=end_date.date()  # Convert to date for the input widget
)

# Convert date_range to datetime objects for comparison
date_range = (
    datetime.combine(date_range[0], datetime.min.time()),
    datetime.combine(date_range[1], datetime.min.time())
)

# Validate and adjust dates if needed
if date_range[1] > end_date:
    date_range = (date_range[0], end_date)
if date_range[0] > date_range[1]:
    date_range = (start_date, end_date)

# Update session state if dates changed
if date_range != st.session_state.date_range:
    st.session_state.date_range = date_range

# Portfolio Parameters
st.sidebar.subheader("Portfolio Parameters")
risk_free_rate = st.sidebar.slider(
    "Risk-free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
) / 100

# Optimization Strategy
optimization_strategy = st.sidebar.selectbox(
    "Optimization Strategy",
    options=[
        "Maximum Sharpe Ratio",
        "Minimum Variance",
        "Maximum Sortino Ratio",
        "Equal Weight",
        "Risk Parity"
    ]
)

# Risk Management Parameters
st.sidebar.subheader("Risk Management")
confidence_level = st.sidebar.slider(
    "VaR Confidence Level (%)",
    min_value=90,
    max_value=99,
    value=95,
    step=1
) / 100

max_position_size = st.sidebar.slider(
    "Maximum Position Size (%)",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=1.0
) / 100

# Initialize data processor and fetch data
@st.cache_data
def load_data(symbols, start_date, end_date):
    processor = DataProcessor(risk_free_rate=risk_free_rate)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    st.write("### Data Loading Information")
    st.write(f"Selected symbols: {symbols}")
    st.write(f"Date range: {start_str} to {end_str}")
    
    try:
        cleaned_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not cleaned_symbols:
            st.error("No valid symbols provided")
            return None, None
            
        st.write(f"Cleaned symbols: {cleaned_symbols}")
        
        prices = processor.fetch_data(cleaned_symbols, start_str, end_str)
        if prices.empty:
            st.error("No data was fetched. Please check your stock symbols and try again.")
            return None, None
            
        st.write(f"Successfully loaded data for {len(prices.columns)} symbols")
        st.write(f"Data shape: {prices.shape}")
        
        returns = processor.calculate_returns(prices)
        if returns.empty:
            st.error("Failed to calculate returns. Please try a different date range.")
            return None, None
            
        st.write(f"Successfully calculated returns. Shape: {returns.shape}")
        return prices, returns
        
    except Exception as e:
        st.error(f"Error during data loading: {str(e)}")
        return None, None

if len(selected_symbols) > 0:
    try:
        prices, returns = load_data(selected_symbols, date_range[0], date_range[1])
        
        if prices is not None and returns is not None and not prices.empty and not returns.empty:
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Optimization", "Risk Analysis", "Technical Analysis", "Performance Metrics"])
            
            with tab1:
                st.header("Portfolio Optimization")
                
                # Initialize portfolio optimizer
                optimizer = PortfolioOptimizer(returns, risk_free_rate)
                
                # Get optimal portfolio based on selected strategy
                strategy_map = {
                    "Maximum Sharpe Ratio": "sharpe",
                    "Minimum Variance": "min_var",
                    "Maximum Sortino Ratio": "sortino",
                    "Equal Weight": "equal",
                    "Risk Parity": "risk_parity"
                }
                
                optimal_portfolio = optimizer.optimize_portfolio(method=strategy_map[optimization_strategy])
                
                # Display portfolio metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Annual Return", f"{optimal_portfolio.returns:.2%}")
                with col2:
                    st.metric("Portfolio Volatility", f"{optimal_portfolio.volatility:.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{optimal_portfolio.sharpe_ratio:.2f}")
                
                # Display optimal weights with position size limits
                st.subheader("Optimal Portfolio Weights")
                weights = optimizer.get_portfolio_weights()
                weights_df = pd.DataFrame(
                    weights.items(),
                    columns=['Asset', 'Weight']
                )
                weights_df['Weight'] = weights_df['Weight'].map('{:.2%}'.format)
                st.dataframe(weights_df)
                
                # Plot efficient frontier
                st.subheader("Efficient Frontier")
                fig = optimizer.plot_efficient_frontier()
                st.plotly_chart(fig, use_container_width=True)
                
                # Display strategy comparison
                st.subheader("Strategy Comparison")
                strategies = ["Maximum Sharpe Ratio", "Minimum Variance", "Maximum Sortino Ratio"]
                comparison_data = []
                
                for strategy in strategies:
                    portfolio = optimizer.optimize_portfolio(method=strategy_map[strategy])
                    comparison_data.append({
                        "Strategy": strategy,
                        "Return": portfolio.returns,
                        "Volatility": portfolio.volatility,
                        "Sharpe Ratio": portfolio.sharpe_ratio
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=comparison_df['Volatility'],
                    y=comparison_df['Return'],
                    mode='markers+text',
                    text=comparison_df['Strategy'],
                    textposition="top center",
                    name="Strategies",
                    marker=dict(size=15)
                ))
                fig.update_layout(
                    title="Strategy Comparison",
                    xaxis_title="Volatility",
                    yaxis_title="Return",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.header("Risk Analysis")
                
                # Calculate and display VaR and CVaR
                processor = DataProcessor(risk_free_rate=risk_free_rate)
                portfolio_weights = optimizer.get_portfolio_weights()
                risk_metrics = processor.calculate_var_cvar(returns, confidence_level, portfolio_weights)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Value at Risk (95%)", f"{risk_metrics['VaR']:.2%}")
                with col2:
                    st.metric("Conditional VaR (95%)", f"{risk_metrics['CVaR']:.2%}")
                
                # Correlation matrix
                st.subheader("Correlation Matrix")
                corr_matrix = returns.corr()
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk decomposition
                st.subheader("Risk Decomposition")
                risk_contribution = processor.calculate_risk_contribution(returns, portfolio_weights)
                fig = px.pie(
                    values=risk_contribution.values(),
                    names=risk_contribution.keys(),
                    title="Risk Contribution by Asset"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.header("Technical Analysis")
                
                # Add technical indicators
                indicators = processor.add_technical_indicators(prices)
                
                # Select asset for technical analysis
                selected_asset = st.selectbox("Select Asset for Technical Analysis", selected_symbols)
                
                # Plot price and indicators
                fig = go.Figure()
                
                # Add price
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=prices[selected_asset],
                    name="Price",
                    line=dict(color="blue")
                ))
                
                # Add moving averages
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=indicators[f'{selected_asset}_SMA_20'],
                    name="20-day SMA",
                    line=dict(color="orange")
                ))
                
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=indicators[f'{selected_asset}_SMA_50'],
                    name="50-day SMA",
                    line=dict(color="green")
                ))
                
                # Add Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=indicators[f'{selected_asset}_BB_High'],
                    name="Upper BB",
                    line=dict(color="gray", dash="dash")
                ))
                
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=indicators[f'{selected_asset}_BB_Low'],
                    name="Lower BB",
                    line=dict(color="gray", dash="dash")
                ))
                
                fig.update_layout(
                    title=f"Technical Analysis - {selected_asset}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display RSI
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=indicators[f'{selected_asset}_RSI'],
                    name="RSI",
                    line=dict(color="purple")
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="red")
                fig.add_hline(y=30, line_dash="dash", line_color="green")
                fig.update_layout(
                    title=f"RSI - {selected_asset}",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.header("Performance Metrics")
                
                # Calculate cumulative returns
                cum_returns = (1 + returns).cumprod()
                
                # Plot cumulative returns
                fig = go.Figure()
                for column in cum_returns.columns:
                    fig.add_trace(go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns[column],
                        name=column,
                        mode='lines'
                    ))
                
                fig.update_layout(
                    title="Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Returns",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display performance metrics
                st.subheader("Asset Performance Metrics")
                metrics_df = pd.DataFrame({
                    'Annual Return': returns.mean() * 252,
                    'Annual Volatility': returns.std() * np.sqrt(252),
                    'Sharpe Ratio': (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252)),
                    'Sortino Ratio': (returns.mean() * 252 - risk_free_rate) / (returns[returns < 0].std() * np.sqrt(252)),
                    'Max Drawdown': (cum_returns / cum_returns.cummax() - 1).min()
                })
                st.dataframe(metrics_df.style.format({
                    'Annual Return': '{:.2%}',
                    'Annual Volatility': '{:.2%}',
                    'Sharpe Ratio': '{:.2f}',
                    'Sortino Ratio': '{:.2f}',
                    'Max Drawdown': '{:.2%}'
                }))
        else:
            st.error("Unable to load data. Please try different stock symbols or date range.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.warning("Please select at least one asset to analyze.") 