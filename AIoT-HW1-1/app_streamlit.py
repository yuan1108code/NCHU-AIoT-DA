import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import altair as alt

# Page layout: Set wide layout to make use of full screen width
st.set_page_config(layout="wide", page_title="Interactive Linear Regression")

# Add custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #F5F5F5;
    }
    .block-container {
        padding: 1.5rem 1.5rem 2rem 1.5rem;
    }
    .stButton button {
        background-color: #007ACC;
        color: white;
        padding: 0.6rem 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit Title with a clear description
st.title("📊 Interactive Linear Regression with Dynamic Data")
st.markdown("""
    This app allows you to explore how different linear regression models perform 
    by dynamically adjusting the slope, noise level, and number of data points. 
    Use the sliders on the left to interact with the model.
""")

# Sidebar for user input
st.sidebar.header("Adjust Parameters")
st.sidebar.markdown("Use the sliders below to customize the data and model behavior.")
a = st.sidebar.slider('Slope (a)', -10.0, 10.0, 3.0)
c = st.sidebar.slider('Noise Level (c)', 0.0, 100.0, 10.0)
n = st.sidebar.slider('Number of Points (n)', 10, 1000, 100)

# Step 2: Generate the data based on user input
np.random.seed(42) 
X = 2 * np.random.rand(n, 1) 
noise = np.random.randn(n, 1) 
y = a * X + 50 + c * noise 

# Display the generated equation and show the data
with st.container():
    st.subheader("🔍 Generated Data Overview")
    st.markdown(f"The data is generated using the equation: `y = {a} * X + 50 + {c} * random_noise`.")
    df = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})
    st.write("Here are the first few rows of the generated data:")
    st.dataframe(df.head(), use_container_width=True)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
st.subheader("🚀 Training the Linear Regression Model")
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict using the test data
y_pred_train = model.predict(X_train) 
y_pred_test = model.predict(X_test)

# Step 6: Visualize the true vs predicted regression lines
st.subheader("📈 Model Visualization")

# Create regression line and true line for comparison
X_line = np.linspace(0, 2, 100).reshape(100, 1)
y_true_line = a * X_line + 50
y_pred_line = model.predict(X_line)

# Create DataFrames for Altair plotting
df_test = pd.DataFrame({'X_test': X_test.flatten(), 'y_test': y_test.flatten(), 'y_pred_test': y_pred_test.flatten()})
line_data = pd.DataFrame({'X_line': X_line.flatten(), 'y_true_line': y_true_line.flatten(), 'y_pred_line': y_pred_line.flatten()})

# Visualize using Altair
true_points_test = alt.Chart(df_test).mark_point(color='blue').encode(
    x=alt.X('X_test', title='X (Test Data)'),
    y=alt.Y('y_test', title='y (True Data)'),
    tooltip=['X_test', 'y_test']
)

predicted_points_test = alt.Chart(df_test).mark_point(color='red').encode(
    x='X_test',
    y='y_pred_test',
    tooltip=['X_test', 'y_pred_test']
)

regression_line = alt.Chart(line_data).mark_line(color='green').encode(
    x='X_line',
    y='y_pred_line'
)

combined_chart = true_points_test + predicted_points_test + regression_line
st.altair_chart(combined_chart, use_container_width=True)

# Step 7: Display evaluation metrics
st.subheader("📊 Model Evaluation Metrics")
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Show metrics in two columns for better readability
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Training MSE", value=f"{mse_train:.2f}")
    st.metric(label="Test MSE", value=f"{mse_test:.2f}")

with col2:
    st.metric(label="Training R-squared (R²)", value=f"{r2_train:.2f}")
    st.metric(label="Test R-squared (R²)", value=f"{r2_test:.2f}")

# Step 9: Display test data with predictions in a table
st.subheader("🔍 Test Data with Predictions")
st.dataframe(df_test, use_container_width=True)

# Optional line chart to visualize test data and predictions
st.line_chart(df_test[['X_test', 'y_test', 'y_pred_test']])