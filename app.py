from flask import Flask, render_template, jsonify, request, send_file, redirect, url_for
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import Error
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from io import BytesIO
import csv

app = Flask(__name__)
CORS(app)

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
    print("Random Forest model loaded successfully")
    
    # Load Isolation Forest for transaction monitoring
    isolation_model = joblib.load('isoforest_model.pkl')
    print("Isolation Forest model loaded successfully")
    
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

def init_db():
    """Verify database connection"""
    try:
        connection = get_pg_connection()
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
        conn = get_pg_connection()
        if conn is None:
            return False
            
        cur = conn.cursor()
        cur.execute('SELECT 1 FROM monitored_transactions WHERE transaction_id = %s', (transaction_id,))
        result = cur.fetchone() is not None
        cur.close()
        conn.close()
        return result
    except Exception as e:
        print(f"Error checking transaction: {e}")
        return False

def store_transaction(transaction):
    """Store a transaction in PostgreSQL database"""
    try:
        conn = get_pg_connection()
        if conn is None:
            raise Exception("Could not connect to database")
            
        cur = conn.cursor()
        
        # Check if transaction already exists
        cur.execute(
            "SELECT transaction_id FROM monitored_transactions WHERE transaction_id = %s",
            (transaction['transaction_id'],)
        )
        
        if cur.fetchone() is None:
            # Insert new transaction
            cur.execute("""
                INSERT INTO monitored_transactions (
                    transaction_id, customer_id, transaction_time, 
                    amount, transaction_type, transaction_description, 
                    is_suspicious
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                transaction['transaction_id'],
                transaction['customer_id'],
                transaction['transaction_time'],
                float(transaction['amount'].replace(',', '')) if isinstance(transaction['amount'], str) else float(transaction['amount']),
                transaction['transaction_type'],
                transaction['transaction_description'],
                bool(transaction['is_suspicious'])
            ))
            
            conn.commit()
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error storing transaction: {str(e)}")
        raise e

def load_data():
    """Load the dataset"""
    try:
        print("\nLoading transaction data from CSV...")
        # Load CSV file with explicit encoding
        df = pd.read_csv("C:/Users/Radhika/.vscode/AML/bank_transactions_trimmed.csv", encoding='utf-8')
        
        # Clean column names (remove any whitespace)
        df.columns = df.columns.str.strip()
        
        print("CSV Columns found:", df.columns.tolist())
        
        # Ensure required columns exist
        required_columns = [
            'transaction_id', 
            'customer_id', 
            'transaction_amount', 
            'transaction_type', 
            'transaction_description',
            'transaction_time'
        ]
        if not all(col in df.columns for col in required_columns):
            print("Error: Missing required columns in CSV file")
            print("Expected columns:", required_columns)
            print("Found columns:", df.columns.tolist())
            return pd.DataFrame()
        
        # Clean and format data
        df['amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        
        # Drop any rows with invalid data
        df = df.dropna(subset=['amount', 'transaction_time'])
        
        print(f"Successfully loaded {len(df)} transactions from CSV")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return pd.DataFrame()

def get_latest_transactions():
    """Get 10 unprocessed transactions from the dataset"""
    # Load fresh data
    df = load_data()
    
    if df.empty:
        print("No data loaded from CSV")
        return pd.DataFrame(), df
    
    print(f"Loaded {len(df)} total transactions")
    
    # Get all processed transaction IDs from the database
    try:
        conn = get_pg_connection()
        if conn:
            cur = conn.cursor()
            cur.execute('SELECT transaction_id FROM monitored_transactions')
            processed_ids = {str(row[0]) for row in cur.fetchall()}
            cur.close()
            conn.close()
            print(f"Found {len(processed_ids)} processed transactions in database")
        else:
            processed_ids = set()
            print("No database connection")
    except Exception as e:
        print(f"Error getting processed transactions: {str(e)}")
        processed_ids = set()
    
    # Convert transaction_id to string for comparison
    df['transaction_id'] = df['transaction_id'].astype(str)
    
    # Filter out already processed transactions
    unprocessed_df = df[~df['transaction_id'].isin(processed_ids)]
    print(f"Found {len(unprocessed_df)} unprocessed transactions")
    
    if len(unprocessed_df) == 0:
        print("No new transactions to process")
        return pd.DataFrame(), df
    
    # Get 10 unprocessed transactions
    latest_transactions = unprocessed_df.head(10).copy()
    print(f"Selected {len(latest_transactions)} new transactions to display")
    
    return latest_transactions, df

def calculate_features(customer_id, pg_connection):
    """Calculate derived features for the ML model"""
    cursor = None
    try:
        print(f"\n=== Calculating features for customer {customer_id} ===")
        cursor = pg_connection.cursor()
        
        # Get customer data
        print("\nFetching customer data...")
        cursor.execute('''
            SELECT annual_income, age
            FROM customers 
            WHERE customer_id = %s
        ''', (customer_id,))
        customer_data = cursor.fetchone()
        
        if not customer_data:
            print(f" No customer found with ID: {customer_id}")
            return None
            
        annual_income, age = customer_data
        # Convert Decimal to float for calculations
        annual_income = float(annual_income) if annual_income else 0.0
        print(f" Found customer: income={annual_income}, age={age}")
        
        # Get transaction data
        print("\nFetching transaction data...")
        cursor.execute('''
            SELECT COUNT(*) as total_transactions,
                   SUM(transaction_amount) as total_amount,
                   AVG(transaction_amount) as avg_amount,
                   MAX(transaction_amount) as max_amount,
                   COUNT(DISTINCT transaction_type) as unique_types
            FROM transactions 
            WHERE customer_id = %s
        ''', (customer_id,))
        transaction_stats = cursor.fetchone()
        
        if not transaction_stats:
            print(f" No transactions found for customer {customer_id}")
            return None
            
        total_transactions, total_amount, avg_amount, max_amount, unique_types = transaction_stats
        # Convert Decimal values to float
        total_amount = float(total_amount) if total_amount else 0.0
        avg_amount = float(avg_amount) if avg_amount else 0.0
        max_amount = float(max_amount) if max_amount else 0.0
        print(f" Found transactions: count={total_transactions}, total={total_amount}")
        
        # Get transaction dates
        cursor.execute('''
            SELECT 
                MIN(transaction_date),
                MAX(transaction_date)
            FROM transactions 
            WHERE customer_id = %s
        ''', (customer_id,))
        date_result = cursor.fetchone()
        first_date, last_date = date_result if date_result else (None, None)
        
        # Calculate derived features
        days_since_first = 0
        if first_date and last_date:
            days_since_first = (last_date - first_date).days
        
        # Calculate transaction frequency
        transaction_frequency = total_transactions / days_since_first if days_since_first > 0 else 0.0
        
        # Calculate amount to income ratio (both values are now float)
        amount_to_income_ratio = total_amount / annual_income if annual_income > 0 else 0.0
        
        # Create features dictionary with exact names expected by model
        features = {
            'annual_income': annual_income,
            'total_transactions': float(total_transactions),
            'total_amount_spent': total_amount,
            'average_transaction_amount': avg_amount,
            'max_transaction_amount': max_amount,
            'unique_transaction_descriptions': float(unique_types),
            'days_since_first_transaction': float(days_since_first),
            'transaction_frequency': float(transaction_frequency),
            'transaction_amount_to_income_ratio': float(amount_to_income_ratio),
            'age': float(age or 0)
        }
        
        print("\n Successfully calculated all features:")
        for name, value in features.items():
            print(f"{name}: {value}")
        
        return features
        
    except Exception as e:
        print(f" Error in calculate_features: {str(e)}")
        import traceback
        print(f"Full error:\n{traceback.format_exc()}")
        return None
    finally:
        if cursor:
            cursor.close()

def check_transaction_dates(customer_id, pg_connection):
    """Check transaction dates format"""
    try:
        cursor = pg_connection.cursor()
        cursor.execute('''
            SELECT transaction_time, transaction_amount, transaction_type
            FROM transactions
            WHERE customer_id = %s
            LIMIT 5
        ''', (customer_id,))
        sample_transactions = cursor.fetchall()
        
        print(f"Sample transactions for customer {customer_id}:")
        for txn in sample_transactions:
            print(f"Time: {txn[0]}, Amount: {txn[1]}, Type: {txn[2]}")
            
        return sample_transactions
    except Exception as e:
        print(f"Error checking transaction dates: {e}")
        return None

def extract_features(transaction, df):
    """Extract relevant features for Isolation Forest model"""
    try:
        # 1. transaction_amount: The amount of the transaction
        amount = float(str(transaction["amount"]).replace(',', ''))
        
        # Get transaction time and customer transactions
        txn_time = pd.to_datetime(transaction["transaction_time"])
        customer_transactions = df[df['customer_id'] == transaction['customer_id']].copy()
        customer_transactions['transaction_time'] = pd.to_datetime(customer_transactions['transaction_time'])
        customer_transactions['amount'] = customer_transactions['amount'].apply(lambda x: float(str(x).replace(',', '')))
        
        # Filter transactions before current transaction for historical analysis
        historical_transactions = customer_transactions[customer_transactions['transaction_time'] < txn_time]
        
        if historical_transactions.empty:
            print(f"\nNo historical transactions found for customer {transaction['customer_id']}")
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
        
        # 2. transaction_type_encoded: Encode transaction type
        transaction_type_mapping = {
            'deposit': 0,
            'payment': 1,
            'transfer': 2,
            'withdrawal': 3
        }
        transaction_type_encoded = transaction_type_mapping.get(transaction["transaction_type"].lower(), 0)
        
        # 3. transaction_description_encoded: Encode description
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
        transaction_description_encoded = transaction_description_mapping.get(transaction["transaction_description"], 0)
        
        # 4. transaction_count_7d: Count of transactions in last 7 days
        seven_days_ago = txn_time - pd.Timedelta(days=7)
        recent_transactions = historical_transactions[historical_transactions['transaction_time'] >= seven_days_ago]
        transaction_count_7d = len(recent_transactions)
        
        # 5. days_since_last_txn: Days since last transaction
        last_txn_time = historical_transactions['transaction_time'].max()
        days_since_last_txn = (txn_time - last_txn_time).days
        
        # Calculate 30-day and 7-day windows
        thirty_days_ago = txn_time - pd.Timedelta(days=30)
        recent_30d = historical_transactions[historical_transactions['transaction_time'] >= thirty_days_ago]
        recent_7d = historical_transactions[historical_transactions['transaction_time'] >= seven_days_ago]
        
        # 6. txn_std_30d: Standard deviation of amounts in last 30 days
        txn_std_30d = recent_30d['amount'].std() if not recent_30d.empty else 0
        
        # 7. txn_mean_7d: Mean transaction amount in last 7 days
        txn_mean_7d = recent_7d['amount'].mean() if not recent_7d.empty else 0
        
        # 8. txn_mean_30d: Mean transaction amount in last 30 days
        txn_mean_30d = recent_30d['amount'].mean() if not recent_30d.empty else 0
        
        # 9. txn_ratio: Current amount vs 30-day mean
        txn_ratio = amount / txn_mean_30d if txn_mean_30d != 0 else 1
        
        # Debug information
        print(f"\nFeature values for transaction {transaction['transaction_id']}:")
        print(f"1. Amount: {amount}")
        print(f"2. Type: {transaction['transaction_type']} -> {transaction_type_encoded}")
        print(f"3. Description: {transaction['transaction_description']} -> {transaction_description_encoded}")
        print(f"4. Transaction count 7d: {transaction_count_7d}")
        print(f"5. Days since last: {days_since_last_txn}")
        print(f"6. Std 30d: {txn_std_30d}")
        print(f"7. Mean 7d: {txn_mean_7d}")
        print(f"8. Mean 30d: {txn_mean_30d}")
        print(f"9. Amount/30d mean ratio: {txn_ratio}")
        
        return [
            amount,                    # 1. transaction_amount
            transaction_type_encoded,  # 2. transaction_type_encoded
            transaction_description_encoded,  # 3. transaction_description_encoded
            transaction_count_7d,      # 4. transaction_count_7d
            days_since_last_txn,      # 5. days_since_last_txn
            txn_std_30d,              # 6. txn_std_30d
            txn_mean_7d,              # 7. txn_mean_7d
            txn_mean_30d,             # 8. txn_mean_30d
            txn_ratio                 # 9. txn_ratio
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
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

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
        print("\nProcessing get-transactions request...")
        latest_transactions, df = get_latest_transactions()
        
        if latest_transactions.empty:
            print("No new transactions found to display")
            return jsonify({'transactions': [], 'message': 'No new transactions'})
        
        print(f"Processing {len(latest_transactions)} new transactions")
        
        # Process each transaction with Isolation Forest
        latest_transactions = predict_suspicious_transactions(latest_transactions, df)
        
        # Format transactions for response
        transactions_list = []
        for _, transaction in latest_transactions.iterrows():
            transaction_dict = {
                'transaction_id': str(transaction['transaction_id']),
                'customer_id': str(transaction['customer_id']),
                'transaction_time': transaction.get('transaction_time', datetime.now().strftime('%H:%M:%S')),
                'amount': transaction['amount'],
                'transaction_type': transaction['transaction_type'],
                'transaction_description': transaction['transaction_description'],
                'is_suspicious': int(transaction.get('is_suspicious', 0))
            }
            
            print(f"Storing transaction {transaction_dict['transaction_id']}")
            # Store the transaction
            store_transaction(transaction_dict)
            
            # Format amount for display
            transaction_dict['amount'] = f"{float(transaction_dict['amount']):,.2f}"
            transactions_list.append(transaction_dict)
        
        print(f"Successfully processed {len(transactions_list)} transactions")
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
        
        # Create features DataFrame
        features_df = pd.DataFrame([features])[feature_order]
        
        # Print features for debugging
        print("\nFeatures for risk prediction:")
        for col in feature_order:
            print(f"{col}: {features_df[col].iloc[0]}")
        
        # Make prediction using Random Forest model
        if risk_model is None:
            return jsonify({'error': 'Risk prediction model not available'}), 500
        
        # Scale numerical features if needed
        scaler = StandardScaler()
        numerical_features = [
            'annual_income',
            'total_amount_spent',
            'average_transaction_amount',
            'max_transaction_amount',
            'transaction_frequency',
            'transaction_amount_to_income_ratio'
        ]
        features_df[numerical_features] = scaler.fit_transform(features_df[numerical_features])
        
        # Get model prediction
        risk_score = risk_model.predict(features_df)[0]
        risk_proba = risk_model.predict_proba(features_df)[0]
        
        print(f"\nRisk prediction: {risk_score}")
        print(f"Risk probabilities: {risk_proba}")
        
        # Map risk categories to levels and colors
        risk_mapping = {
            0: {'level': 'Low Risk', 'color': '#00C851', 'description': 'Normal transaction patterns'},
            1: {'level': 'Medium Risk', 'color': '#ffbb33', 'description': 'Some unusual patterns detected'},
            2: {'level': 'High Risk', 'color': '#ff4444', 'description': 'Significant anomalies detected'}
        }
        
        risk_info = risk_mapping.get(risk_score, {'level': 'Unknown', 'color': '#ffffff', 'description': 'Unable to determine risk'})
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'risk_score': int(risk_score),
            'risk_level': risk_info['level'],
            'risk_color': risk_info['color'],
            'risk_description': risk_info['description'],
            'risk_probabilities': risk_proba.tolist(),
            'features': features
        })
        
    except Exception as e:
        print(f"Error calculating risk score: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route("/select-report")
def select_report():
    """Display report type selection page"""
    return render_template('select_report.html')

@app.route("/generate-pdf")
def generate_pdf():
    """Generate PDF report of suspicious transactions"""
    try:
        # Connect to database
        conn = get_pg_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Fetch suspicious transactions
        cursor.execute("""
            SELECT transaction_id, customer_id, transaction_time, amount, 
                   transaction_type, transaction_description
            FROM monitored_transactions 
            WHERE is_suspicious = true 
            ORDER BY transaction_time DESC
        """)
        
        suspicious_transactions = cursor.fetchall()
        
        if not suspicious_transactions:
            return jsonify({'message': 'No suspicious transactions found'}), 404
        
        # Create PDF using ReportLab
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Define table data
        data = [['Transaction ID', 'Customer ID', 'Time', 'Amount', 'Type', 'Description']]
        for transaction in suspicious_transactions:
            data.append([
                str(transaction[0]),
                str(transaction[1]),
                str(transaction[2]),
                f"${transaction[3]:,.2f}",
                transaction[4],
                transaction[5]
            ])
        
        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        # Prepare response
        buffer.seek(0)
        cursor.close()
        conn.close()
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'suspicious_transactions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
        return jsonify({'error': str(e)}), 500

@app.route("/generate-csv")
def generate_csv():
    """Generate CSV report of suspicious transactions"""
    try:
        # Connect to database
        conn = get_pg_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Fetch suspicious transactions
        cursor.execute("""
            SELECT transaction_id, customer_id, transaction_time, amount, 
                   transaction_type, transaction_description
            FROM monitored_transactions 
            WHERE is_suspicious = true 
            ORDER BY transaction_time DESC
        """)
        
        suspicious_transactions = cursor.fetchall()
        
        if not suspicious_transactions:
            return jsonify({'message': 'No suspicious transactions found'}), 404
        
        # Create CSV in memory
        output = BytesIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Transaction ID', 'Customer ID', 'Time', 'Amount', 'Type', 'Description'])
        
        # Write data
        for transaction in suspicious_transactions:
            writer.writerow([
                str(transaction[0]),
                str(transaction[1]),
                str(transaction[2]),
                f"${transaction[3]:,.2f}",
                transaction[4],
                transaction[5]
            ])
        
        # Prepare response
        output.seek(0)
        cursor.close()
        conn.close()
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'suspicious_transactions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        print(f"Error generating CSV report: {str(e)}")
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
