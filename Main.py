import yfinance as yf  #Importing `yfinance` library to fetch historical stock price data from Yahoo Finance.
import pandas as pd  #Importing `pandas` for data manipulation and analysis.
import plotly.graph_objects as go  #Importing `plotly.graph_objects` for creating interactive plots.
from statsmodels.tsa.arima.model import ARIMA  #Importing `ARIMA` model from `statsmodels` for time series forecasting.
from sklearn.metrics import mean_squared_error, mean_absolute_error  #Importing metrics from `sklearn` to evaluate model performance.
import numpy as np  #Importing `numpy` for numerical operations.
from prophet import Prophet  #Importing `Prophet` from `prophet` for advanced time series forecasting.
import streamlit as st  #Importing `streamlit` to create a web interface for the application.

#Set Streamlit page configuration to define the app's layout, title, and icon.
st.set_page_config(
    page_title='Advanced Stock Price Prediction',  #Page title for the Streamlit app.
    page_icon='📈',  #Page icon for the Streamlit app.
    layout='wide',  #Setting the layout to wide for better visualization.
    initial_sidebar_state='expanded'  #Setting the sidebar to be expanded initially.
)

#Fetch stock data function using yfinance and cache the data for efficient reuse.
@st.cache_data
def fetch_stock_data(ticker):
    data = yf.download(ticker, start='2010-01-01', end='2023-01-01')  #Downloading historical stock price data from Yahoo Finance.
    data = data[['Close']]  #Selecting the 'Close' column from the fetched data.
    data.reset_index(inplace=True)  #Resetting index to make 'Date' a column.
    return data  #Returning the processed data.

#ARIMA model function to predict stock prices and evaluate the model.
def arima_model(data, p, d, q):
    train_size = int(len(data) * 0.8)  #Defining the training size as 80% of the data.
    train, test = data['Close'][:train_size], data['Close'][train_size:]  #Splitting the data into training and testing sets.

    model = ARIMA(train, order=(p, d, q))  #Creating an ARIMA model with specified order.
    model_fit = model.fit()  #Fitting the ARIMA model on the training data.
    forecast = model_fit.forecast(steps=len(test))  #Forecasting future prices for the length of the test data.
    forecast = np.array(forecast)  #Converting forecast to a numpy array.

    mse = mean_squared_error(test, forecast)  #Calculating Mean Squared Error to evaluate the forecast.
    mae = mean_absolute_error(test, forecast)  #Calculating Mean Absolute Error to evaluate the forecast.
    return forecast, test, mse, mae  #Returning forecast, test data, and evaluation metrics.

#Prophet model function to predict stock prices using the Prophet library.
def prophet_model(data):
    prophet_data = data.rename(columns={'Date': 'ds', 'Close': 'y'})  #Renaming columns for compatibility with Prophet.
    model_prophet = Prophet()  #Creating a Prophet model instance.
    model_prophet.fit(prophet_data)  #Fitting the Prophet model on the data.

    future = model_prophet.make_future_dataframe(periods=365)  #Creating a dataframe to hold future dates for prediction.
    forecast_prophet = model_prophet.predict(future)  #Predicting future prices using the Prophet model.
    return forecast_prophet  #Returning the forecasted data.

#Streamlit web interface for user interaction and visualization.
st.title('Advanced Stock Price Prediction')  #Title of the Streamlit app.

#Introductory Information about the app.
st.markdown("""
Welcome to the Advanced Stock Price Prediction app! This app allows you to predict future stock prices using two powerful models: ARIMA and Prophet. Whether you're new to the stock market or an experienced trader, this app provides insights to help you make informed decisions.

**How to Use This App:**
1. Enter a stock ticker symbol (e.g., 'AAPL' for Apple).
2. View the historical stock price data.
3. Choose model parameters and run the ARIMA model to forecast future prices.
4. Use the Prophet model to predict stock prices.
5. Explore the stock's performance summary and apply moving averages to smooth out the data.
""")  #Detailed description of the app's purpose and usage.

ticker = st.text_input('Enter Stock Ticker', 'TSLA')  #Text input widget for entering the stock ticker symbol.
data = fetch_stock_data(ticker)  #Fetching stock data based on the entered ticker symbol.

if not data.empty:  #Checking if data is not empty before proceeding.
    #Stock Data Summary section.
    st.subheader(f'{ticker} Stock Price Data')  #Subheader for stock price data section.
    st.write(f'The data below shows the historical closing prices for {ticker}.')  #Description of the data.
    
    fig = go.Figure()  #Creating a new plotly figure.
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))  #Adding a line trace for closing prices.
    fig.update_layout(title=f'{ticker} Stock Price Data', xaxis_title='Date', yaxis_title='Close Price')  #Updating layout with titles.
    st.plotly_chart(fig, use_container_width=True)  #Displaying the plot in the Streamlit app.

    #Sidebar for ARIMA model parameters.
    st.sidebar.subheader('ARIMA Model Parameters')  #Subheader for ARIMA model parameters in the sidebar.
    st.sidebar.markdown("""
    **ARIMA Model:**
    - ARIMA (AutoRegressive Integrated Moving Average) is a popular time series forecasting model.
    - Choose the AR (p), differencing (d), and MA (q) parameters to run the model.
    """)  #Description of ARIMA model.

    p = st.sidebar.slider('p (AR order)', 1, 5, 1)  #Slider to select AR order.
    d = st.sidebar.slider('d (differencing order)', 0, 2, 1)  #Slider to select differencing order.
    q = st.sidebar.slider('q (MA order)', 0, 5, 1)  #Slider to select MA order.

    if st.sidebar.button('Run ARIMA Model'):  #Button to run the ARIMA model.
        forecast_arima, test, mse_arima, mae_arima = arima_model(data, p, d, q)  #Running the ARIMA model with selected parameters.
        
        st.subheader('ARIMA Model Prediction')  #Subheader for ARIMA model prediction results.
        st.write(f'Test MSE (Mean Squared Error): {mse_arima}')  #Displaying Mean Squared Error.
        st.write(f'Test MAE (Mean Absolute Error): {mae_arima}')  #Displaying Mean Absolute Error.
        st.write(f'The ARIMA model has predicted future prices based on the historical data with the following errors.')  #Description of prediction errors.

        fig_arima = go.Figure()  #Creating a new plotly figure for ARIMA predictions.
        fig_arima.add_trace(go.Scatter(x=data['Date'][len(test):], y=test, mode='lines', name='Actual'))  #Adding actual test data trace.
        fig_arima.add_trace(go.Scatter(x=data['Date'][len(test):], y=forecast_arima, mode='lines', name='Forecast'))  #Adding forecast data trace.
        fig_arima.update_layout(title='ARIMA Model Prediction', xaxis_title='Date', yaxis_title='Price')  #Updating layout with titles.
        st.plotly_chart(fig_arima, use_container_width=True)  #Displaying the ARIMA prediction plot.

    #Sidebar for Prophet model.
    st.sidebar.subheader('Prophet Model')  #Subheader for Prophet model in the sidebar.
    st.sidebar.markdown("""
    **Prophet Model:**
    - Prophet is a forecasting model developed by Facebook, particularly effective for time series with daily observations.
    - Click the button below to run the Prophet model.
    """)  #Description of Prophet model.

    if st.sidebar.button('Run Prophet Model'):  #Button to run the Prophet model.
        forecast_prophet = prophet_model(data)  #Running the Prophet model.
        
        st.subheader('Prophet Model Prediction')  #Subheader for Prophet model prediction results.
        st.write(f'The Prophet model has forecasted the future prices based on historical data.')  #Description of Prophet prediction.

        fig_prophet = go.Figure()  #Creating a new plotly figure for Prophet predictions.
        fig_prophet.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual'))  #Adding actual data trace.
        fig_prophet.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat'], mode='lines', name='Forecast'))  #Adding forecast data trace.
        fig_prophet.update_layout(title='Prophet Model Prediction', xaxis_title='Date', yaxis_title='Price')  #Updating layout with titles.
        st.plotly_chart(fig_prophet, use_container_width=True)  #Displaying the Prophet prediction plot.

    #Sidebar for Data Overview.
    st.sidebar.subheader('Data Overview')  #Subheader for data overview in the sidebar.
    st.sidebar.markdown("""
    **Data Overview:**
    - View the summary statistics of the stock data including mean, standard deviation, min, and max values.
    """)  #Description of data overview section.
    st.sidebar.write(data.describe())  #Displaying summary statistics of the stock data.

    #Sidebar for Feature Engineering.
    st.sidebar.subheader('Feature Engineering')  #Subheader for feature engineering in the sidebar.
    st.sidebar.markdown("""
    **Moving Average:**
    - Apply a moving average to smooth out the stock price data.
    - Choose a window size for the moving average calculation.
    """)  #Description of moving average feature.

    window_size = st.sidebar.slider('Moving Average Window Size', 1, 50, 5)  #Slider to select moving average window size.
    data['MA'] = data['Close'].rolling(window=window_size).mean()  #Calculating moving average based on selected window size.

    if st.sidebar.button('Apply Moving Average'):  #Button to apply the moving average.
        st.subheader('Moving Average')  #Subheader for moving average section.
        st.write(f'The moving average with a window size of {window_size} days is applied to the stock price data.')  #Description of moving average application.

        fig_ma = go.Figure()  #Creating a new plotly figure for moving average.
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))  #Adding actual close price trace.
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA'], mode='lines', name='Moving Average'))  #Adding moving average trace.
        fig_ma.update_layout(title='Moving Average', xaxis_title='Date', yaxis_title='Price')  #Updating layout with titles.
        st.plotly_chart(fig_ma, use_container_width=True)  #Displaying the moving average plot.

    #Summary of Stock Performance section.
    st.subheader('Stock Performance Summary')  #Subheader for stock performance summary section.
    st.write(f'Here is a brief summary of {ticker} stock performance:')  #Description of stock performance summary.
    summary = data.describe()  #Calculating summary statistics of the stock data.
    st.write(summary)  #Displaying the summary statistics.

    st.markdown("""
    **Key Metrics:**
    - **Mean Price:** The average closing price of the stock.
    - **Standard Deviation:** Indicates the volatility of the stock.
    - **Min/Max Prices:** The lowest and highest closing prices observed in the given period.
    """)  #Explanation of key metrics in the stock performance summary.
