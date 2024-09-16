import pandas as pd
import mysql.connector
from mysql.connector import Error
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import plotly.express as px
import plotly.utils
import json
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import joblib
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path='/assets', static_folder='assets')
app.secret_key = "admin8074"

# Load the trained model
model = joblib.load('joblibs/xgboost_sales_forecast_model.joblib')

# Extract feature names from the model
# Since we're dealing with RandomizedSearchCV, we need to access the best estimator
best_model = model.best_estimator_
# Assuming the XGBoost model is the last step in the pipeline
xgb_model = best_model.named_steps['xgb']
feature_names = xgb_model.get_booster().feature_names

# Print feature names for debugging
print("Feature names:", feature_names)

# Create and fit a LabelEncoder for sales_cat
le = LabelEncoder()
le.fit(['Corporate', 'Retail', 'Show Rooms'])  # Make sure this matches your categories

# Function to create database connection
def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection

# Function to execute query
def execute_query(connection, query, params=None):
    cursor = connection.cursor()
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")
    finally:
        cursor.close()

# Function to get yearly sales data for the past 5 years
def get_yearly_sales_data():
    connection = create_db_connection("localhost", "root", "", "sales_predic_sys")
    current_year = datetime.now().year
    query = f"""
    SELECT sales_cat, SUM(net_value) as yearly_sales, year
    FROM sales_data 
    WHERE year BETWEEN {current_year - 5} AND {current_year - 1}
    GROUP BY sales_cat, year
    ORDER BY year, sales_cat
    """
    result = execute_query(connection, query)
    connection.close()
    
    df = pd.DataFrame(result, columns=['sales_cat', 'yearly_sales', 'year'])
    return df

# Function to get last month's individual sales data
def get_last_month_data():
    connection = create_db_connection("localhost", "root", "", "sales_predic_sys")
    query = f"""
    SELECT sales_cat, SUM(net_value) AS net_value,  DATE_FORMAT(STR_TO_DATE(CONCAT(year, '-', month), '%Y-%M'), '%Y-%m') AS maxdate
    FROM sales_data
    WHERE DATE_FORMAT(STR_TO_DATE(CONCAT(year, '-', month), '%Y-%M'), '%Y-%m') = (SELECT MAX(DATE_FORMAT(STR_TO_DATE(CONCAT(year, '-', month), '%Y-%M'), '%Y-%m')) FROM sales_data)
    GROUP BY sales_cat
    """
    result = execute_query(connection, query)
    connection.close()
    
    df = pd.DataFrame(result, columns=['sales_cat', 'net_value', 'maxdate'])
    return df
    
def format_number(num):
    return f"{num:,}"

# Decorator to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in to access this page.')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/assets/<path:path>')
def send_asset(path):
    return send_from_directory('assets', path)

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        connection = create_db_connection("localhost", "root", "", "sales_predic_sys")
        
        # Check if username already exists
        check_query = "SELECT * FROM users WHERE username = %s"
        result = execute_query(connection, check_query, (username,))
        
        if result:
            flash('Username already exists. Please choose a different one.')
            connection.close()
            return redirect(url_for('register'))
        
        # Insert new user
        hashed_password = generate_password_hash(password)
        insert_query = "INSERT INTO users (username, password) VALUES (%s, %s)"
        try:
            cursor = connection.cursor()
            cursor.execute(insert_query, (username, hashed_password))
            connection.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except Error as err:
            print(f"Error: '{err}'")
            flash('Registration failed. Please try again.')
        finally:
            cursor.close()
            connection.close()
    
    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        connection = create_db_connection("localhost", "root", "", "sales_predic_sys")
        query = "SELECT * FROM users WHERE username = %s"
        result = execute_query(connection, query, (username,))
        connection.close()
        
        if result and check_password_hash(result[0][2], password):
            session['logged_in'] = True
            session['username'] = username
            flash('You have been logged in!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Index route (dashboard)
@app.route('/')
@login_required
def index():
    df = get_yearly_sales_data()
    
    # Create a line chart for all categories over the past 5 years
    fig = px.line(df, x='year', y='yearly_sales', color='sales_cat',
                  title='Yearly Sales Trends by Category (Past 5 Years)',
                  labels={'yearly_sales': 'Sales Value', 'year': 'Year', 'sales_cat': 'Sales Category'})

    fig.update_layout(
        legend_title_text='Sales Category',
        xaxis_title='Year',
        yaxis_title='Sales Value',
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        xaxis=dict(
            tickmode='linear',
            tick0=df['year'].min(),
            dtick=1,
            tickformat='d'
        )
    )

    fig.update_traces(line=dict(width=2))

    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    last_month_data = get_last_month_data()
    card_data = {row['sales_cat']: {'value': format_number(int(row['net_value']))} for _, row in last_month_data.iterrows()}
    maxdate = last_month_data['maxdate'].iloc[0] if not last_month_data.empty else None

    return render_template('index.html', chart_json=chart_json, card_data=card_data, maxdate=maxdate)

def prepare_input_data(year, category):
    # Create a DataFrame with the input data
    input_df = pd.DataFrame({
        'year': [year],
        'month': ['January'],  # You might want to allow the user to input the month as well
        'sales_cat': [category]
    })
    
    # Preprocess input data
    input_df['date'] = pd.to_datetime(input_df['year'].astype(str) + '-' + input_df['month'], format='%Y-%B')
    input_df['month_num'] = input_df['date'].dt.month
    input_df['year_num'] = input_df['date'].dt.year
    input_df['quarter'] = input_df['date'].dt.quarter
    input_df['is_holiday_season'] = ((input_df['month_num'] == 11) | (input_df['month_num'] == 12)).astype(int)
    input_df['days_in_month'] = input_df['date'].dt.days_in_month
    
    # Encode the sales_cat
    input_df['sales_cat_encoded'] = le.transform(input_df['sales_cat'])
    
    # Add other necessary features (you might need to adjust these based on your actual model)
    input_df['sales_growth'] = 0  # You may want to use an average or recent growth rate
    input_df['cumulative_sales'] = 0  # You may want to calculate this based on historical data
    input_df['sales_ma_3'] = 0
    input_df['sales_ma_6'] = 0
    input_df['sales_ma_12'] = 0
    
    # Add lag features and rolling mean features
    for lag in [1, 3, 6, 12]:
        input_df[f'net_value_lag_{lag}'] = 0
    for window in [3, 6, 12]:
        input_df[f'net_value_rolling_mean_{window}'] = 0
    
    # Create a new DataFrame with the correct feature order
    input_data = pd.DataFrame(columns=feature_names)
    for feature in feature_names:
        if feature in input_df.columns:
            input_data[feature] = input_df[feature]
        else:
            input_data[feature] = 0  # or some appropriate default value

    return input_data

def predict_net_value_with_intervals(input_data, percentile=95):
    # Make prediction using the best estimator
    prediction = model.best_estimator_.predict(input_data)
    
    # Calculate intervals (you might need to adjust this based on your actual model and data)
    # For this example, we'll use a simple method. In practice, you might want to use a more sophisticated approach.
    std_resid = 1000000  # This is a placeholder. You should calculate this based on your model's performance on test data.
    z_score = norm.ppf((1 + percentile/100) / 2)
    interval = z_score * std_resid
    lower_bound = prediction - interval
    upper_bound = prediction + interval
    
    return prediction[0], max(0, lower_bound[0]), upper_bound[0]

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        category = request.form['category']
        
        # Prepare input data for prediction
        input_data = prepare_input_data(year, category)
        
        # Make prediction
        prediction, lower_bound, upper_bound = predict_net_value_with_intervals(input_data)
        
        return render_template('predict.html', prediction=prediction, lower_bound=lower_bound, upper_bound=upper_bound, year=year, category=category)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)