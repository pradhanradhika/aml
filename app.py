from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import psycopg2
from psycopg2 import Error
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load the ML model
try:
    print("\nLoading ML model...")
    model = joblib.load('random_forest_model.pkl')
    print("✅ ML model loaded successfully")
    
    # Validate model features
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
    
    # Check if model has the expected number of features
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        if len(model_features) != len(expected_features):
            print(f"❌ ERROR: Model expects {len(model_features)} features but we're providing {len(expected_features)}")
            model = None
        else:
            print("✅ Model feature count validated")
    else:
        print("⚠️ Warning: Model does not have feature names, assuming correct order")
        
except Exception as e:
    print(f"❌ Error loading ML model: {e}")
    import traceback
    print(f"Full traceback:\n{traceback.format_exc()}")
    model = None

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
    'password': 'radhika@28',  # Change this to your PostgreSQL password
    'database': 'aml'   # Change this to your database name
}

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
        print("✅ Successfully connected to PostgreSQL database")
        return connection
    except Error as e:
        print(f"\n❌ ERROR connecting to PostgreSQL Database: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR connecting to PostgreSQL Database: {str(e)}")
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

def store_transaction(transaction, is_suspicious):
    """Store a processed transaction in the database"""
    try:
        connection = get_db_connection()
        if connection is None:
            return
            
        cursor = connection.cursor()
        current_timestamp = datetime.now()
        cursor.execute('''INSERT INTO processed_transactions 
                         (transaction_id, customer_id, transaction_time, amount, 
                          transaction_type, description, is_suspicious, processed_at)
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                      (transaction['transaction_id'], transaction['customer_id'],
                       transaction['transaction_time'], float(transaction['amount'].replace(',', '')),
                       transaction['transaction_type'], transaction['description'],
                       int(is_suspicious), current_timestamp))
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        print(f"Error storing transaction: {e}")

def load_data():
    """Load the dataset"""
    return pd.read_csv("bank_transactions_trimmed.csv")

def get_latest_transactions():
    """Get 10 random transactions from the dataset"""
    # Load fresh data
    df = load_data()
    print("Total transactions loaded:", len(df))
    
    # Get 10 random transactions
    latest_transactions = df.sample(n=10, random_state=None)  # random_state=None ensures different selection each time
    print("Selected transactions:", len(latest_transactions))
    
    # Keep original transaction_time for ML features
    # Use current_time for display
    latest_transactions["display_time"] = datetime.now().strftime("%H:%M:%S")
    
    return latest_transactions, df

def extract_features(transaction, df):
    """Extract relevant features for ML model using dataset transaction times"""
    # Get the transaction amount
    amount = float(str(transaction["transaction_amount"]).replace(',', ''))
    
    # Get transaction time from dataset
    txn_time = datetime.strptime(transaction["transaction_time"], "%H:%M:%S")
    
    # Calculate transaction count in last 7 days using dataset times
    seven_days_ago = txn_time - pd.Timedelta(days=7)
    transaction_count_7d = len(df[df["transaction_time"].apply(
        lambda x: datetime.strptime(x, "%H:%M:%S") >= seven_days_ago
    )])
    
    # Calculate days since last transaction using dataset times
    last_txn = df[df["transaction_time"].apply(
        lambda x: datetime.strptime(x, "%H:%M:%S") < txn_time
    )].sort_values("transaction_time", ascending=False)
    
    days_since_last_txn = 0
    if not last_txn.empty:
        last_txn_time = datetime.strptime(last_txn.iloc[0]["transaction_time"], "%H:%M:%S")
        days_since_last_txn = (txn_time - last_txn_time).days
    
    # Calculate transaction statistics using dataset times
    thirty_days_ago = txn_time - pd.Timedelta(days=30)
    seven_days_ago = txn_time - pd.Timedelta(days=7)
    
    txn_std_30d = df[df["transaction_time"].apply(
        lambda x: datetime.strptime(x, "%H:%M:%S") >= thirty_days_ago
    )]["transaction_amount"].std()
    
    txn_mean_7d = df[df["transaction_time"].apply(
        lambda x: datetime.strptime(x, "%H:%M:%S") >= seven_days_ago
    )]["transaction_amount"].mean()
    
    txn_mean_30d = df[df["transaction_time"].apply(
        lambda x: datetime.strptime(x, "%H:%M:%S") >= thirty_days_ago
    )]["transaction_amount"].mean()
    
    # Calculate transaction ratio (7d mean / 30d mean)
    txn_ratio = txn_mean_7d / txn_mean_30d if txn_mean_30d != 0 else 0
    
    # Transaction type encoding mapping
    transaction_type_mapping = {
        'deposit': 0,
        'payment': 1,
        'transfer': 2,
        'withdrawal': 3
    }
    
    # Transaction description encoding mapping
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
    
    # Get encoded values using the mappings
    transaction_type_encoded = transaction_type_mapping.get(transaction["transaction_type"], 0)  # Default to 0 if not found
    transaction_description_encoded = transaction_description_mapping.get(transaction["transaction_description"], 0)  # Default to 0 if not found
    
    return [
        amount,                    # transaction_amount
        transaction_type_encoded,  # transaction_type_encoded
        transaction_description_encoded,  # transaction_description_encoded
        transaction_count_7d,      # transaction_count_7d
        days_since_last_txn,       # days_since_last_txn
        txn_std_30d,              # txn_std_30d
        txn_mean_7d,              # txn_mean_7d
        txn_mean_30d,             # txn_mean_30d
        txn_ratio                 # txn_ratio
    ]

def predict_suspicious_transactions(transactions, df):
    """Pass transactions to ML model and get predictions"""
    features = [extract_features(t, df) for _, t in transactions.iterrows()]
    predictions = model.predict(features)  # Predict using ML model
    transactions["is_suspicious"] = predictions  # Add predictions column
    return transactions

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
    try:
        latest_transactions, df = get_latest_transactions()
        
        # Get predictions for new transactions using dataset times
        latest_transactions = predict_suspicious_transactions(latest_transactions, df)
        
        # Convert to list of dictionaries for JSON response
        transactions_list = []
        
        # Process transactions in the exact order they appear
        for _, row in latest_transactions.iterrows():
            transaction = {
                'customer_id': str(row["customer_id"]),
                'transaction_id': str(row["transaction_id"]),
                'transaction_time': row['display_time'],  # Use current time for display
                'amount': f'{row["transaction_amount"]:,.2f}',
                'transaction_type': row['transaction_type'],
                'description': row['transaction_description'],
                'is_suspicious': int(row["is_suspicious"])
            }
            
            # Add to list regardless of processing status
            transactions_list.append(transaction)
            
            # Only store if not already processed
            if not is_transaction_processed(transaction['transaction_id']):
                # Store with original transaction time from dataset
                store_transaction({
                    **transaction,
                    'transaction_time': row['transaction_time']  # Use original time for storage
                }, transaction['is_suspicious'])
                print("Added transaction:", transaction['transaction_id'])
            else:
                print("Transaction already processed:", transaction['transaction_id'])
        
        print("Total transactions in response:", len(transactions_list))
        return jsonify(transactions_list)
    except Exception as e:
        print("Error in get_transactions:", str(e))
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
    try:
        customer_id = request.form.get('customer_id')
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400

        connection = get_pg_connection()
        if connection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Verify customer exists
        cursor = connection.cursor()
        cursor.execute('''
            SELECT customer_id, name, annual_income, age
            FROM customers 
            WHERE customer_id = %s
        ''', (customer_id,))
        customer_data = cursor.fetchone()
        
        if not customer_data:
            return jsonify({'error': 'Customer not found'}), 404
        
        # Check if customer has transactions
        cursor.execute('''
            SELECT COUNT(*), 
                   MIN(transaction_date),
                   MAX(transaction_date)
            FROM transactions 
            WHERE customer_id = %s
        ''', (customer_id,))
        txn_count, first_txn, last_txn = cursor.fetchone()
        
        if txn_count == 0:
            return jsonify({'error': 'No transactions found for this customer'}), 400
        
        # Calculate features
        features = calculate_features(customer_id, connection)
        if features is None:
            return jsonify({'error': 'Could not calculate features for this customer'}), 400

        # Convert features to DataFrame for model prediction
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
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'ML model not available'}), 500
            
        risk_category = model.predict(features_df)[0]
        
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
        print(f"Error in calculate_risk_score: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Initialize the database when the app starts
    init_db()
    app.run(debug=True)
