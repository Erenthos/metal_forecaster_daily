import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Title of the app
st.title('MCX Copper Price Forecasting with Variance (using RandomForestRegressor Algorithm)')

# Sample data (you can replace this with actual data)
data = {
    'Day': ['28 Oct ', '29 Oct', '30 Oct', '31 Oct', '1 Nov', '4 2024', '5 2024'],
    'Actual_Price': [795.40, 806.20, 803.70, 814.60, 847.80, 852.70, 857.50]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['Day'] = pd.to_datetime(df['Day'])  # Convert to datetime
df['Day_Number'] = np.arange(1, len(df) + 1)

# Features and target variable
X = df[['Day_Number']]
y = df['Actual_Price']

# Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict prices for historical months
df['Predicted_Price'] = model.predict(X)

# Input: Number of months to forecast
st.sidebar.header('Forecast Settings')
num_months = st.sidebar.slider('Select number of months to forecast:', 1, 4, 3)

# Generate future month labels
last_month = df['Day'].iloc[-1]  # Get the last month in the DataFrame
future_months = [(last_month + pd.DateOffset(months=i)).strftime('%b %Y') for i in range(1, num_months + 1)]

# Predict future prices
future_month_numbers = np.arange(len(df) + 1, len(df) + 1 + num_months).reshape(-1, 1)
predicted_future_prices = model.predict(future_month_numbers)

# Create future dataframe
future_df = pd.DataFrame({
    'Month': future_months,
    'Predicted_Price': predicted_future_prices
})

# Input: Actual values for forecasted months (optional)
st.sidebar.header('Actual Values for Forecasted Months (Optional)')
actual_future_prices = st.sidebar.text_input('Enter actual prices for forecasted months, separated by commas (e.g., 750, 760)', '')

# Process the actual prices if provided
if actual_future_prices:
    actual_future_prices = [float(x) for x in actual_future_prices.split(',')]
    if len(actual_future_prices) == num_months:
        future_df['Actual_Price'] = actual_future_prices
        future_df['Variance (%)'] = ((future_df['Predicted_Price'] - future_df['Actual_Price']) / future_df['Actual_Price']) * 100
    else:
        st.error(f"Please provide {num_months} values for the forecasted months.")

# Display future prices
st.subheader('Forecasted Prices for Upcoming Days')
st.write(future_df)

# Plot actual vs predicted prices
st.subheader('Price Forecast Visualization')
fig, ax = plt.subplots(figsize=(10, 5))

# Plot historical data
ax.plot(df['Month'].dt.strftime('%b %Y'), df['Actual_Price'], label='Actual Price', marker='o')
ax.plot(df['Month'].dt.strftime('%b %Y'), df['Predicted_Price'], label='Predicted Price', linestyle='--', marker='x')

# Plot future predicted prices
ax.plot(future_months, future_df['Predicted_Price'], label='Future Forecast', linestyle='-.', marker='s', color='red')

# Make the x-axis scalable to avoid label overlap
plt.xticks(rotation=45, ha='right')
ax.set_xlabel('Day')
ax.set_ylabel('Price (INR/KG)')
ax.set_title('Actual vs Predicted Prices')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate variance (error percentage) for historical data
df['Variance (%)'] = ((df['Predicted_Price'] - df['Actual_Price']) / df['Actual_Price']) * 100

# Display variance table
st.subheader('Variance between Actual and Predicted Prices (Historical Data)')
st.write(df[['Day', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])

# Display variance table for future data (if actual values were provided)
if 'Variance (%)' in future_df.columns:
    st.subheader('Variance between Actual and Predicted Prices (Forecasted Data)')
    st.write(future_df[['Month', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])
