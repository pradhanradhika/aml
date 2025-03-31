from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import psycopg2
from psycopg2 import Error
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
CORS(app)

# MySQL configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  
    'password': 'Ra@238gs',  
    'database': 'aml'  }

# PostgreSQL configuration
PG_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'radhika@28',  
    'database': 'aml'   
}

# Load the ML models
try:
    print("\nLoading ML models...")
    # Load Random Forest for risk score prediction
    risk_model = joblib.load('random_forest_model.pkl')
    print("✅ Random Forest model loaded successfully")
    
    # Load Isolation Forest for transaction monitoring
    isolation_model = joblib.load('isolation_forest_model (2).pkl')
    print("✅ Isolation Forest model loaded successfully")
    
    # Validate Random Forest model features
    expected_features = [
        'annual_income',
        'total_transactions',
        'total_amount_spent',
        'average_transaction_amount',
        'max_transaction_amount',
        'unique_transaction_descriptions',
        'days_since_first_transaction',
        'transaction_frequency',
        'transaction_amount_to_income_ratio',
        'age'
    ]
    
    # Check if Random Forest model has the expected number of features
    if hasattr(risk_model, 'feature_names_in_'):
        model_features = list(risk_model.feature_names_in_)
        if len(model_features) != len(expected_features):
            print(f"ERROR: Random Forest model expects {len(model_features)} features but we're providing {len(expected_features)}")
            risk_model = None
        else:
            print("Random Forest model feature count validated")
    else:
        print("Warning: Random Forest model does not have feature names, assuming correct order")
        
except Exception as e:
    print(f" Error loading ML models: {e}")
    import traceback
    print(f"Full traceback:\n{traceback.format_exc()}")
    isolation_model = None
    risk_model = None

def get_db_connection():
    """Create a database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None

def get_pg_connection():
    """Create a PostgreSQL database connection"""
    try:
        print("\nAttempting to connect to PostgreSQL database with config:")
        print(f"Host: {PG_CONFIG['host']}")
        print(f"Database: {PG_CONFIG['database']}")
        print(f"User: {PG_CONFIG['user']}")
        
        connection = psycopg2.connect(**PG_CONFIG)
        print("Successfully connected to PostgreSQL database")
        return connection
    except Error as e:
        print(f"\n ERROR connecting to PostgreSQL Database: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None
    except Exception as e:
        print(f"\n UNEXPECTED ERROR connecting to PostgreSQL Database: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None

def init_db():
    """Verify database connection"""
    try:
        connection = get_db_connection()
        if connection:
            connection.close()
            print("Database connection successful")
        else:
            print("Failed to connect to database")
    except Error as e:
        print(f"Error connecting to database: {e}")

def is_transaction_processed(transaction_id):
    """Check if a transaction has already been processed"""
    try:
        connection = get_db_connection()
        if connection is None:
            return False
            
        cursor = connection.cursor()
        cursor.execute('SELECT 1 FROM processed_transactions WHERE transaction_id = %s', (transaction_id,))
        result = cursor.fetchone() is not None
        cursor.close()
        connection.close()
        return result
    except Error as e:
        print(f"Error checking transaction: {e}")
        return False

def store_transaction(transaction):
    """Store a transaction in the database"""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            
            # Check if transaction already exists
            cursor.execute('SELECT transaction_id FROM processed_transactions WHERE transaction_id = %s', 
                         (transaction['transaction_id'],))
            if cursor.fetchone() is None:
                # Insert new transaction
                cursor.execute('''
                    INSERT INTO processed_transactions 
                    (transaction_id, customer_id, transaction_time, amount, transaction_type, description, is_suspicious)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (
                    transaction['transaction_id'],
                    transaction['customer_id'],
                    transaction['transaction_time'],
                    transaction['amount'],  # Already a float
                    transaction['transaction_type'],
                    transaction['transaction_description'],
                    transaction.get('is_suspicious', 0)
                ))
                connection.commit()
            cursor.close()
            connection.close()
    except Error as e:
        print(f"Error storing transaction: {e}")

def load_data():
    """Load the dataset"""
    df = pd.read_csv("bank_transactions_trimmed.csv")
    
    # Convert amount to float and time to datetime with proper format
    df['amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')
    
    # Drop any rows with invalid data
    df = df.dropna(subset=['amount', 'transaction_time'])
    
    return df

def get_latest_transactions():
    """Get 10 unprocessed transactions from the dataset"""
    # Load fresh data
    df = load_data()
    
    # Get all processed transaction IDs from the database
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute('SELECT transaction_id FROM processed_transactions')
            processed_ids = {str(row[0]) for row in cursor.fetchall()}
            cursor.close()
            connection.close()
        else:
            processed_ids = set()
    except Error as e:
        print(f"Error getting processed transactions: {e}")
        processed_ids = set()
    
    # Convert transaction_id to string for comparison
    df['transaction_id'] = df['transaction_id'].astype(str)
    
    # Filter out already processed transactions
    unprocessed_df = df[~df['transaction_id'].isin(processed_ids)]
    
    if len(unprocessed_df) == 0:
        print("No new transactions to process")
        return pd.DataFrame(), df
    
    # Get 10 unprocessed transactions
    latest_transactions = unprocessed_df.head(10).copy()
    
    # Add display time
    latest_transactions["display_time"] = datetime.now().strftime("%H:%M:%S")
    
    return latest_transactions, df

def calculate_features(customer_id, pg_connection):
    """Calculate derived features for the ML model"""
    try:
        print(f"Calculating features for customer: {customer_id}")
        
        cursor = pg_connection.cursor()
        
        # Get customer data
        cursor.execute('''
            SELECT annual_income, age
            FROM customers 
            WHERE customer_id = %s
        ''', (customer_id,))
        customer_data = cursor.fetchone()
        
        if not customer_data:
            print(f"Customer {customer_id} not found in customers table")
            return None
            
        annual_income, age = customer_data
        
        # Get basic transaction stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_transactions,
                SUM(transaction_amount) as total_amount_spent,
                AVG(transaction_amount) as average_transaction_amount,
                MAX(transaction_amount) as max_transaction_amount,
                COUNT(DISTINCT transaction_type) as unique_transaction_descriptions
            FROM transactions 
            WHERE customer_id = %s
        ''', (customer_id,))
        txn_stats = cursor.fetchone()
        
        if not txn_stats or txn_stats[0] == 0:
            print(f"No transactions found for customer {customer_id}")
            return None
            
        total_transactions, total_amount_spent, avg_amount, max_amount, unique_types = txn_stats
        
        # Get transaction dates
        cursor.execute('''
            SELECT 
                MIN(transaction_date),
                MAX(transaction_date)
            FROM transactions 
            WHERE customer_id = %s
        ''', (customer_id,))
        first_date, last_date = cursor.fetchone()
        
        # Calculate derived features
        try:
            # Calculate days between first and last transaction
            days_since_first = 0
            if first_date and last_date:
                days_since_first = (last_date - first_date).days
            
            # Calculate transaction frequency
            transaction_frequency = 0
            if days_since_first > 0:
                transaction_frequency = total_transactions / days_since_first
            
            # Calculate transaction amount to income ratio
            transaction_amount_to_income_ratio = 0
            if annual_income and annual_income > 0:
                transaction_amount_to_income_ratio = total_amount_spent / annual_income
            
            # Create features dictionary
            features = {
                'annual_income': float(annual_income or 0),
                'total_transactions': int(total_transactions or 0),
                'total_amount_spent': float(total_amount_spent or 0),
                'average_transaction_amount': float(avg_amount or 0),
                'max_transaction_amount': float(max_amount or 0),
                'unique_transaction_descriptions': int(unique_types or 0),
                'days_since_first_transaction': int(days_since_first or 0),
                'transaction_frequency': float(transaction_frequency or 0),
                'transaction_amount_to_income_ratio': float(transaction_amount_to_income_ratio or 0),
                'age': int(age or 0)
            }
            
            return features
            
        except Exception as calc_error:
            print(f"Error calculating derived features: {str(calc_error)}")
            return None
        
    except Exception as e:
        print(f"Error in calculate_features: {str(e)}")
        return None

def check_transaction_dates(customer_id, pg_connection):
    """Check transaction dates format"""
    try:
        cursor = pg_connection.cursor()
        cursor.execute('''
            SELECT transaction_date, transaction_amount, transaction_type
            FROM transactions
            WHERE customer_id = %s
            LIMIT 5
        ''', (customer_id,))
        sample_transactions = cursor.fetchall()
        
        print(f"Sample transactions for customer {customer_id}:")
        for txn in sample_transactions:
            print(f"Date: {txn[0]}, Amount: {txn[1]}, Type: {txn[2]}")
            
        return sample_transactions
    except Exception as e:
        print(f"Error checking transaction dates: {e}")
        return None

def extract_features(transaction, df):
    """Extract relevant features for Isolation Forest model"""
    try:
        # Get the transaction amount
        amount = float(str(transaction["amount"]).replace(',', ''))
        
        # Get transaction time
        txn_time = pd.to_datetime(transaction["transaction_time"])
        
        # Get customer's transactions
        customer_transactions = df[df['customer_id'] == transaction['customer_id']].copy()
        customer_transactions['transaction_time'] = pd.to_datetime(customer_transactions['transaction_time'])
        customer_transactions['amount'] = customer_transactions['amount'].apply(lambda x: float(str(x).replace(',', '')))
        
        # Filter transactions before current transaction
        historical_transactions = customer_transactions[customer_transactions['transaction_time'] < txn_time]
        
        if historical_transactions.empty:
            return [
                amount,                    # transaction_amount
                0,                         # transaction_type_encoded
                0,                         # transaction_description_encoded
                0,                         # transaction_count_7d
                0,                         # days_since_last_txn
                0,                         # txn_std_30d
                0,                         # txn_mean_7d
                0,                         # txn_mean_30d
                0                          # txn_ratio
            ]
        
        # Calculate transaction count in last 7 days
        seven_days_ago = txn_time - pd.Timedelta(days=7)
        recent_transactions = historical_transactions[historical_transactions['transaction_time'] >= seven_days_ago]
        transaction_count_7d = len(recent_transactions)
        
        # Calculate days since last transaction
        if not historical_transactions.empty:
            last_txn_time = historical_transactions['transaction_time'].max()
            days_since_last_txn = (txn_time - last_txn_time).days
        else:
            days_since_last_txn = 0
        
        # Calculate transaction statistics
        thirty_days_ago = txn_time - pd.Timedelta(days=30)
        recent_30d = historical_transactions[historical_transactions['transaction_time'] >= thirty_days_ago]
        recent_7d = historical_transactions[historical_transactions['transaction_time'] >= seven_days_ago]
        
        # Calculate statistics
        txn_std_30d = recent_30d['amount'].std() if not recent_30d.empty else 0
        txn_mean_7d = recent_7d['amount'].mean() if not recent_7d.empty else 0
        txn_mean_30d = recent_30d['amount'].mean() if not recent_30d.empty else 0
        
        # Calculate transaction ratio
        txn_ratio = txn_mean_7d / txn_mean_30d if txn_mean_30d != 0 else 0
        
        # Transaction type encoding
        transaction_type_mapping = {
            'deposit': 0,
            'payment': 1,
            'transfer': 2,
            'withdrawal': 3
        }
        
        # Transaction description encoding
        transaction_description_mapping = {
            'ATM Withdrawal': 0,
            'Bank Transfer': 1,
            'Cash Deposit': 2,
            'Cash Withdrawal': 3,
            'Check Deposit': 4,
            'Online Purchase': 5,
            'Online Transaction': 6,
            'Online Transfer': 7,
            'Restaurant Bill': 8,
            'Salary Deposit': 9,
            'Utility Bill': 10
        }
        
        transaction_type_encoded = transaction_type_mapping.get(transaction["transaction_type"].lower(), 0)
        transaction_description_encoded = transaction_description_mapping.get(transaction["transaction_description"], 0)
        
        # Debug information
        print(f"\nFeature values for transaction {transaction['transaction_id']}:")
        print(f"Amount: {amount}")
        print(f"Type: {transaction['transaction_type']} -> {transaction_type_encoded}")
        print(f"Description: {transaction['transaction_description']} -> {transaction_description_encoded}")
        print(f"Transaction count 7d: {transaction_count_7d}")
        print(f"Days since last: {days_since_last_txn}")
        print(f"Std 30d: {txn_std_30d}")
        print(f"Mean 7d: {txn_mean_7d}")
        print(f"Mean 30d: {txn_mean_30d}")
        print(f"Ratio: {txn_ratio}")
        
        return [
            amount,                    # transaction_amount
            transaction_type_encoded,  # transaction_type_encoded
            transaction_description_encoded,  # transaction_description_encoded
            transaction_count_7d,      # transaction_count_7d
            days_since_last_txn,      # days_since_last_txn
            txn_std_30d,              # txn_std_30d
            txn_mean_7d,              # txn_mean_7d
            txn_mean_30d,             # txn_mean_30d
            txn_ratio                 # txn_ratio
        ]
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None

def predict_suspicious_transactions(transactions, df):
    """Use Isolation Forest to predict suspicious transactions"""
    features = []
    for _, transaction in transactions.iterrows():
        transaction_features = extract_features(transaction, df)
        if transaction_features:
            features.append(transaction_features)
    
    if not features:
        return transactions
    
    # Get predictions from Isolation Forest
    predictions = isolation_model.predict(features)
    
    # Debug: Print the predictions
    print("\nPredictions from Isolation Forest:")
    print(f"Raw predictions: {predictions}")
    print(f"Number of suspicious transactions: {sum(predictions == -1)}")
    print(f"Number of normal transactions: {sum(predictions == 1)}")
    
    # Isolation Forest returns -1 for anomalies (suspicious) and 1 for normal transactions
    transactions["is_suspicious"] = (predictions == -1).astype(int)
    
    return transactions

@app.route("/")
def index():
    """Serve the index page"""
    return render_template("index.html")

@app.route("/login")
@app.route("/login.html")
def login():
    """Serve the login page"""
    return render_template("login.html")

@app.route("/monitor-transactions")
def monitor_transactions():
    """Serve the monitor transactions page"""
    return render_template("monitor_transaction.html")

@app.route("/risk-score")
def risk_score():
    """Serve the risk score page"""
    return render_template("risk_score.html")

@app.route("/generate-report")
def generate_report():
    """Serve the generate report page"""
    return render_template("generate_report.html")

@app.route("/get-transactions")
def get_transactions():
    """Get latest transactions"""
    try:
        latest_transactions, df = get_latest_transactions()
        
        if latest_transactions.empty:
            return jsonify({'transactions': [], 'message': 'No new transactions'})
        
        # Process each transaction with Isolation Forest
        latest_transactions = predict_suspicious_transactions(latest_transactions, df)
        
        # Format transactions for response
        transactions_list = []
        for _, transaction in latest_transactions.iterrows():
            transaction_dict = {
                'transaction_id': str(transaction['transaction_id']),
                'customer_id': str(transaction['customer_id']),
                'transaction_time': transaction.get('display_time', transaction['transaction_time'].strftime('%H:%M:%S')),
                'amount': transaction['amount'],  # Keep as float for database
                'transaction_type': transaction['transaction_type'],
                'transaction_description': transaction['transaction_description'],
                'is_suspicious': int(transaction['is_suspicious'])
            }
            
            # Store the transaction
            store_transaction(transaction_dict)
            
            # Format amount for display only after storing
            transaction_dict['amount'] = f"{transaction['amount']:,.2f}"
            transactions_list.append(transaction_dict)
        
        return jsonify({
            'transactions': transactions_list,
            'message': f'Found {len(transactions_list)} new transactions'
        })
        
    except Exception as e:
        print(f"Error in get_transactions: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route("/search-customer", methods=['POST'])
def search_customer():
    try:
        customer_id = request.form.get('customer_id')
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400

        print(f"Attempting to connect to PostgreSQL database...")
        connection = get_pg_connection()
        if connection is None:
            print("Failed to connect to PostgreSQL database")
            return jsonify({'error': 'Database connection failed. Please check your database configuration.'}), 500

        print(f"Successfully connected to database. Searching for customer {customer_id}...")
        cursor = connection.cursor()
        
        # Query to get customer details from existing customers table
        query = '''
            SELECT 
                customer_id,
                name,
                age,
                annual_income,
                city_state,
                email
            FROM customers
            WHERE customer_id = %s
        '''
        print(f"Executing query: {query}")
        cursor.execute(query, (customer_id,))
        
        result = cursor.fetchone()
        
        if result is None:
            print(f"Customer {customer_id} not found")
            return jsonify({'error': 'Customer not found'}), 404
            
        print(f"Found customer: {result[1]}")  # Log customer name
        
        # Format the result
        customer_data = {
            'customer_id': result[0],
            'name': result[1],
            'age': result[2],
            'annual_income': float(result[3]) if result[3] else 0,
            'city_state': result[4],
            'email': result[5]
        }
        
        cursor.close()
        connection.close()
        print("Successfully retrieved customer data")
        
        return jsonify(customer_data)
        
    except Exception as e:
        print(f"Error searching customer: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route("/calculate-risk-score", methods=['POST'])
def calculate_risk_score():
    """Calculate risk score for a customer"""
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400
            
        # Get PostgreSQL connection
        connection = get_pg_connection()
        if not connection:
            return jsonify({'error': 'Failed to connect to database'}), 500
            
        cursor = connection.cursor()
        
        # Calculate features for risk score prediction
        features = calculate_features(customer_id, connection)
        if not features:
            cursor.close()
            connection.close()
            return jsonify({'error': 'Failed to calculate features'}), 500
            
        # Order features as expected by the model
        feature_order = [
            'annual_income',
            'total_transactions',
            'total_amount_spent',
            'average_transaction_amount',
            'max_transaction_amount',
            'unique_transaction_descriptions',
            'days_since_first_transaction',
            'transaction_frequency',
            'transaction_amount_to_income_ratio',
            'age'
        ]
        features_df = pd.DataFrame([features])[feature_order]
        
        # Make prediction using Random Forest model
        if risk_model is None:
            return jsonify({'error': 'Risk prediction model not available'}), 500
            
        risk_category = risk_model.predict(features_df)[0]
        
        # Map risk categories to levels and colors
        risk_mapping = {
            0: {'level': 'Low Risk', 'color': '#00C851'},
            1: {'level': 'Medium Risk', 'color': '#ffbb33'},
            2: {'level': 'High Risk', 'color': '#ff4444'}
        }
        
        risk_info = risk_mapping.get(risk_category, {'level': 'Unknown', 'color': '#ffffff'})
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'risk_score': int(risk_category),
            'risk_level': risk_info['level'],
            'risk_color': risk_info['color'],
            'features': features
        })
        
    except Exception as e:
        print(f"Error calculating risk score: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Initialize the database when the app starts
    init_db()
    app.run(debug=True)
