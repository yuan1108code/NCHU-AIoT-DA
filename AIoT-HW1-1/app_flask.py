import os
import time
import traceback
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

# Ensure static folder exists to save generated plots
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def index():
    """Render the main HTML page with user input form."""
    return render_template('index.html')

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    try:
        # Get user inputs from the form
        n = int(request.form['num_points'])  # Number of points
        a = float(request.form['slope'])    # Slope
        c = float(request.form['noise'])    # Noise level
        b = 50  # Intercept for the linear equation

        # Step 1: Generate data
        np.random.seed(42)
        X = 2 * np.random.rand(n, 1)  # Generate random data for X
        noise = np.random.randn(n, 1)  # Noise
        y = a * X + b + c * noise  # Linear equation with noise

        # Step 2: Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 3: Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 4: Predict on both train and test data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Step 5: Prepare evaluation metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Step 6: Generate a plot of the data and predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Test Data (True)')
        plt.plot(X_test, y_pred_test, color='red', label='Predicted Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Linear Regression: Slope={a}, Noise={c}, Points={n}')
        plt.legend()

        # Save the plot with a unique name to avoid caching issues
        timestamp = int(time.time())  # Generate a unique timestamp
        plot_path = f'static/regression_plot_{timestamp}.png'
        plt.savefig(plot_path)
        plt.close()

        # Return JSON response with plot and evaluation metrics
        return jsonify({
            'plot_url': plot_path,
            'mse_train': round(mse_train, 2),
            'mse_test': round(mse_test, 2),
            'r2_train': round(r2_train, 2),
            'r2_test': round(r2_test, 2)
        })
    except Exception as e:
        print("Error occurred:", str(e))
        print(traceback.format_exc())  # Log the stack trace
        return jsonify({'error': str(e)}), 500  # Return error

if __name__ == '__main__':
    app.run(debug=True, port=5050)