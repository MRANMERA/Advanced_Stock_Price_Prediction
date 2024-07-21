# Advanced Stock Price Prediction App

Welcome to the **Advanced Stock Price Prediction** app! This interactive application leverages advanced models to predict future stock prices based on historical data. It uses sophisticated techniques and visualizations to help you understand stock trends and make informed decisions. Imagine an app that predicts stock prices like a crystal ball, but uses real data and fancy math! This advanced stock price prediction app lets you explore future possibilities for a stock's price. Enter a ticker symbol, and the app fetches historical data to get a sense of how the stock has behaved.

## How to Use This App:

1. Enter a stock ticker symbol (e.g., 'AAPL' for Apple).
2. View the historical stock price data.
3. Choose model parameters and run the ARIMA model to forecast future prices.
4. Use the Prophet model to predict stock prices.
5. Explore the stock's performance summary and apply moving averages to smooth out the data.

## Libraries Used

This app integrates several powerful libraries:

- **`yfinance`**: Fetches historical stock price data from Yahoo Finance.
- **`pandas`**: Facilitates data manipulation and analysis.
- **`plotly.graph_objects`**: Creates interactive and dynamic visualizations.
- **`statsmodels`**: Provides the ARIMA model for time series forecasting.
- **`sklearn.metrics`**: Evaluates forecasting accuracy using metrics like MSE and MAE.
- **`numpy`**: Supports numerical operations and array handling.
- **`prophet`**: Offers Prophet, a forecasting model by Facebook for time series data.
- **`streamlit`**: Builds the web app interface for user interaction.

## User-Friendly Interface

The app leverages Streamlit to create a user-friendly and interactive interface. Here's a breakdown of its key functionalities:

### Stock Ticker Input
- Users can effortlessly enter a stock ticker symbol (e.g., AAPL for Apple) to initiate the analysis.

### Historical Data Visualization
- Upon receiving the ticker symbol, the app retrieves historical closing prices using yfinance. This data is then visualized with Plotly, allowing users to explore historical trends and price movements.

### Forecasting Powerhouse

#### ARIMA Model
- This section empowers users to delve into ARIMA (AutoRegressive Integrated Moving Average) forecasting. Users can define parameters like:
  - **p (AR order)**: This controls the number of past closing prices considered for predictions.
  - **d (differencing order)**: This accounts for trends and seasonality in the data.
  - **q (MA order)**: This incorporates the influence of past forecast errors on future predictions.

#### Prophet Model
- This section features the Prophet model, particularly effective for forecasting time series with daily observations. Users can simply click a button to leverage its capabilities.

### Interactive Visualizations
- The app displays the predictions from both ARIMA and Prophet alongside the actual historical data. This visualization, powered by Plotly, enables users to compare and contrast the models' forecasts.

### In-depth Data Exploration

#### Stock Performance Summary
- This section provides a detailed overview of the stock's performance, including metrics like mean price, standard deviation, minimum, and maximum closing prices.

#### Moving Averages
- Users can calculate and visualize a moving average for the stock price data. This helps smooth out short-term fluctuations and identify broader trends. The window size for the moving average can be customized by the user.

## Code Breakdown (Conceptual)

### 1. Setting the Stage
- The code begins by importing necessary libraries like `yfinance`, `pandas`, `plotly.graph_objects`, `statsmodels.tsa.arima.model`, `sklearn.metrics`, `numpy`, and `prophet`. It then utilizes `st.set_page_config` to configure the Streamlit app's appearance, including title, icon, layout, and initial sidebar state.

### 2. Fetching Stock Data
- The `fetch_stock_data` function retrieves historical closing prices for a specified ticker symbol (`ticker`) using `yfinance`. The data is then filtered to include only the "Close" price and reset for proper indexing.

### 3. ARIMA Model Implementation
- The `arima_model` function takes the stock data, p, d, and q values (AR, differencing, and MA orders) as input. Here's a breakdown of its steps:
  - **Splitting Data**: The data is divided into training and testing sets.
  - **Model Fitting**: An ARIMA model is created using the training data with the specified parameters.
  - **Forecasting**: The model forecasts prices for the testing set.
  - **Error Calculation**: Mean Squared Error (MSE) and Mean Absolute Error (MAE) are calculated to evaluate the model's accuracy.

### 4. Prophet Model Integration
- The `prophet_model` function prepares the data for Prophet by renaming columns and converting it to the required format. It then fits a Prophet model to the data and generates forecasts for future periods.

### 5. Streamlit Web Interface Construction

The app utilizes Streamlit to create an engaging and interactive interface. Below is a comprehensive breakdown of each Streamlit component and its functionality:

### Title
- **`st.title('Advanced Stock Price Prediction')`**
  - Sets the main title of the app, providing a clear and concise introduction to the purpose of the application.

### Introductory Information
- **`st.markdown(""" ... """)`**
  - Displays a markdown block that welcomes users and provides instructions on how to use the app. This section explains the purpose of the app, how to enter stock ticker symbols, and what to expect from the analysis.

### Stock Ticker Input
- **`ticker = st.text_input('Enter Stock Ticker', 'TSLA')`**
  - Creates a text input field where users can enter the ticker symbol of the stock they want to analyze. The default value is set to 'TSLA' (Tesla).

### Historical Data Visualization
- **`if not data.empty: ...`**
  - Checks if the fetched data is not empty. If data is available, it displays:
  - **`st.subheader(f'{ticker} Stock Price Data')`**
    - Provides a subheading for the historical data section.
  - **`st.write(f'The data below shows the historical closing prices for {ticker}.')`**
    - Shows a brief description of the data being displayed.
  - **`fig = go.Figure() ... st.plotly_chart(fig, use_container_width=True)`**
    - Uses Plotly to create a line chart showing historical closing prices. The chart is then rendered in the app for interactive exploration.

### ARIMA Model Parameters
- **`st.sidebar.subheader('ARIMA Model Parameters')`**
  - Provides a subheading in the sidebar for ARIMA model configuration.
- **`st.sidebar.markdown(""" ... """)`**
  - Displays explanatory text about the ARIMA model and its parameters.
- **`p = st.sidebar.slider('p (AR order)', 1, 5, 1)`**
  - A slider for selecting the AR (AutoRegressive) order.
- **`d = st.sidebar.slider('d (differencing order)', 0, 2, 1)`**
  - A slider for selecting the differencing order.
- **`q = st.sidebar.slider('q (MA order)', 0, 5, 1)`**
  - A slider for selecting the MA (Moving Average) order.
- **`if st.sidebar.button('Run ARIMA Model'): ...`**
  - Runs the ARIMA model when the button is clicked, displays predictions, and visualizes them alongside actual data.

### Prophet Model Integration
- **`st.sidebar.subheader('Prophet Model')`**
  - Provides a subheading in the sidebar for the Prophet model section.
- **`st.sidebar.markdown(""" ... """)`**
  - Displays explanatory text about the Prophet model.
- **`if st.sidebar.button('Run Prophet Model'): ...`**
  - Runs the Prophet model when the button is clicked, displays predictions, and visualizes them alongside actual data.

### Data Overview
- **`st.sidebar.subheader('Data Overview')`**
  - Provides a subheading in the sidebar for the data summary.
- **`st.sidebar.markdown(""" ... """)`**
  - Displays explanatory text about the data overview.
- **`st.sidebar.write(data.describe())`**
  - Shows summary statistics of the stock data, including mean, standard deviation, minimum, and maximum values.

### Feature Engineering
- **`st.sidebar.subheader('Feature Engineering')`**
  - Provides a subheading in the sidebar for feature engineering.
- **`st.sidebar.markdown(""" ... """)`**
  - Displays explanatory text about moving averages.
- **`window_size = st.sidebar.slider('Moving Average Window Size', 1, 50, 5)`**
  - A slider for selecting the window size for the moving average calculation.
- **`if st.sidebar.button('Apply Moving Average'): ...`**
  - Applies the moving average with the selected window size, visualizes it, and displays the results.

### Stock Performance Summary
- **`st.subheader('Stock Performance Summary')`**
  - Provides a subheading for the stock performance summary.
- **`st.write(f'Here is a brief summary of {ticker} stock performance:')`**
  - Displays a brief summary of the stock's performance.
- **`summary = data.describe()`**
  - Calculates summary statistics of the stock data.
- **`st.write(summary)`**
  - Shows the summary statistics in the app.
