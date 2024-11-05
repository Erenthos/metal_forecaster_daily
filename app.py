import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Title of the app
st.title('MCX Copper Price Forecasting with Variance (using RandomForestRegressor Algorithm)')

# Sample data for one week (you can replace this with actual data)
data = {
    'Date': pd.date_range(start='2023-10-28', periods=7, freq='D'),  # Daily values for a week
    'Actual_Price': [795.40, 806.20, 803.70, 814.60, 847.80, 852.70, 857.50]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['Day_Number'] = np.arange(1, len(df) + 1)  # Convert to sequential day number

# Features and target variable
X = df[['Day_Number']]
y = df['Actual_Price']

# Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict prices for historical days
df['Predicted_Price'] = model.predict(X)

# Input: Number of days to forecast
st.sidebar.header('Forecast Settings')
num_days = st.sidebar.slider('Select number of days to forecast:', 1, 7, 3)  # Adjust for daily forecasting

# Generate future day labels
last_date = df['Date'].iloc[-1]  # Get the last date in the DataFrame
future_dates = [(last_date + pd.DateOffset(days=i)).strftime('%Y-%m-%d') for i in range(1, num_days + 1)]

# Predict future prices
future_day_numbers = np.arange(len(df) + 1, len(df) + 1 + num_days).reshape(-1, 1)
predicted_future_prices = model.predict(future_day_numbers)

# Create future dataframe
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': predicted_future_prices
})

# Input: Actual values for forecasted days (optional)
st.sidebar.header('Actual Values for Forecasted Days (Optional)')
actual_future_prices = st.sidebar.text_input('Enter actual prices for forecasted days, separated by commas (e.g., 750, 760)', '')

# Process the actual prices if provided
if actual_future_prices:
    actual_future_prices = [float(x) for x in actual_future_prices.split(',')]
    if len(actual_future_prices) == num_days:
        future_df['Actual_Price'] = actual_future_prices
        future_df['Variance (%)'] = ((future_df['Predicted_Price'] - future_df['Actual_Price']) / future_df['Actual_Price']) * 100
    else:
        st.error(f"Please provide {num_days} values for the forecasted days.")

# Display future prices
st.subheader('Forecasted Prices for Upcoming Days')
st.write(future_df)

# Plot actual vs predicted prices
st.subheader('Price Forecast Visualization')
fig, ax = plt.subplots(figsize=(10, 5))

# Plot historical data
ax.plot(df['Date'].dt.strftime('%Y-%m-%d'), df['Actual_Price'], label='Actual Price', marker='o')
ax.plot(df['Date'].dt.strftime('%Y-%m-%d'), df['Predicted_Price'], label='Predicted Price', linestyle='--', marker='x')

# Plot future predicted prices
ax.plot(future_dates, future_df['Predicted_Price'], label='Future Forecast', linestyle='-.', marker='s', color='red')

# Make the x-axis scalable to avoid label overlap
plt.xticks(rotation=45, ha='right')
ax.set_xlabel('Date')
ax.set_ylabel('Price (INR/KG)')
ax.set_title('Actual vs Predicted Prices')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate variance (error percentage) for historical data
df['Variance (%)'] = ((df['Predicted_Price'] - df['Actual_Price']) / df['Actual_Price']) * 100

# Display variance table
st.subheader('Variance between Actual and Predicted Prices (Historical Data)')
st.write(df[['Date', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])

# Display variance table for future data (if actual values were provided)
if 'Variance (%)' in future_df.columns:
    st.subheader('Variance between Actual and Predicted Prices (Forecasted Data)')
    st.write(future_df[['Date', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])
