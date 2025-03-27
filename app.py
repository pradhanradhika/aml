from flask import Flask, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
CORS(app)

# Load the ML model
ml_model = joblib.load("isolation_forest_model (1).pkl")

# MySQL configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Change this to your MySQL username
    'password': 'Ra@238gs',  # Change this to your MySQL password
    'database': 'aml'  # Change this to your desired database name
}


def get_db_connection():
    """Create a database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
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
    predictions = ml_model.predict(features)  # Predict using ML model
    transactions["is_suspicious"] = predictions  # Add predictions column
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

if __name__ == "__main__":
    # Initialize the database when the app starts
    init_db()
    app.run(debug=True)
