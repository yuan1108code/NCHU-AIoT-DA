# Interactive Linear Regression App

This repository contains two versions of an interactive linear regression web application:
1. **Flask-based Linear Regression App** (`app_flask.py`)
2. **Streamlit-based Linear Regression App** (`app_streamlit.py`)

Both apps allow users to dynamically generate synthetic data for linear regression, train a model, and visualize the results, with the ability to adjust parameters such as slope, noise level, and the number of data points.

## Table of Contents

- [Interactive Linear Regression App](#interactive-linear-regression-app)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Install Dependencies](#install-dependencies)
    - [Flask-based App](#flask-based-app)
        - [Features of Flask App](#features-of-flask-app)
    - [Streamlit-based App](#streamlit-based-app)
        - [Features of Streamlit App](#features-of-streamlit-app)
    - [Conclusion](#conclusion)
  
## Setup

### Prerequisites

To run both the Flask and Streamlit apps, you need to have Python installed on your system. You'll also need to install the required packages, which are common between both apps.

### Install Dependencies

Run the following command to install all required packages:
```bash
pip install -r requirements.txt
```

### Flask-based App

The Flask-based app (app_flask.py) allows users to adjust parameters for generating synthetic data and visualize the linear regression model directly from a web page. The app uses Flask for the backend and matplotlib for generating plots.

How to Run Flask App
1.	Open a terminal and navigate to the directory containing app_flask.py.
2.	Run the Flask app using the following command:
    ```python
    python app_flask.py
    ```
3.	Open a web browser and go to http://localhost:5050/.

##### Features of Flask App

- **Dynamic Input**: Users can adjust the slope, noise level, and number of points using a form.
- **Interactive Visualization**: The app generates and displays a scatter plot of the data and regression line.
- **Prediction Evaluation**: After fitting the model, the app displays key evaluation metrics such as Mean Squared Error (MSE) and R-squared (R²).
- **Customizable Parameters**:
    1. Slope (a): Controls the slope of the generated data.
    2. Noise (c): Controls the level of noise added to the data.
    3. Number of Points (n): Controls the number of data points in the dataset.

### Streamlit-based App

The Streamlit-based app (app_streamlit.py) is an easy-to-use, interactive web application that allows users to visualize and adjust the parameters for generating synthetic data in real time.

How to Run Streamlit App
1. Open a terminal and navigate to the directory containing app_streamlit.py.
2. Run the Streamlit app using the following command:
    ```bash
    streamlit run app_streamlit.py
    ```
3. Open a web browser and go to http://localhost:8501/.

##### Features of Streamlit App
- **User-Friendly Interface**: Streamlit’s simple interface allows users to dynamically adjust parameters such as slope, noise level, and number of points with sliders.
- **Real-Time Data Visualization**: The app displays the generated data points, regression line, and predictions in real-time using interactive charts.
- **Evaluation Metrics**: The app computes and displays evaluation metrics such as Training and Test Mean Squared Error (MSE) and R-squared (R²).
- **Customizable Parameters**:
  1. Slope (a): Adjust the slope of the linear equation.
  2. Noise Level (c): Adjust the noise level to simulate randomness in the data.
  3. Number of Points (n): Choose how many data points to generate for the model.


### Conclusion

Both the Flask and Streamlit apps provide a dynamic way to explore how linear regression models behave with different data inputs. Choose the Flask app if you need a more traditional web app setup, or go with Streamlit for a more interactive, data science-focused experience.

Feel free to use or modify these apps for your learning, projects, or demonstrations.