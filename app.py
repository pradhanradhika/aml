# app.py
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
from database import get_pg_connection, store_transaction
from transactions import get_latest_transactions
from MLmodels import load_models
from features import calculate_features, extract_features
from reports import generate_pdf, generate_csv
from config import EXPECTED_FEATURES

app = Flask(__name__)
CORS(app)

# Load ML models
risk_model, isolation_model = load_models()

def predict_suspicious_transactions(transactions, df):
    """Use Isolation Forest to predict suspicious transactions"""
    features = []
    for _, transaction in transactions.iterrows():
        transaction_features = extract_features(transaction, df)
        if transaction_features:
            features.append(transaction_features)
    
    if not features:
        return transactions
    
    predictions = isolation_model.predict(features)
    print("\nPredictions from Isolation Forest:")
    print(f"Raw predictions: {predictions}")
    print(f"Number of suspicious transactions: {sum(predictions == -1)}")
    print(f"Number of normal transactions: {sum(predictions == 1)}")
    
    transactions["is_suspicious"] = (predictions == -1).astype(int)
    return transactions

@app.route("/")
def index():
    return redirect(url_for('login'))

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/dashboard", methods=['GET', 'POST'])
def dashboard():
    return render_template("index.html")

@app.route("/monitor-transactions")
def monitor_transactions():
    return render_template("monitor_transaction.html")

@app.route("/risk-score")
def risk_score():
    return render_template("risk_score.html")

@app.route("/generate-report")
def generate_report():
    return render_template("generate_report.html")

@app.route("/get-transactions")
def get_transactions():
    try:
        print("\nProcessing get-transactions request...")
        latest_transactions, df = get_latest_transactions()
        
        if latest_transactions.empty:
            print("No new transactions found to display")
            return jsonify({'transactions': [], 'message': 'No new transactions'})
        
        print(f"Processing {len(latest_transactions)} new transactions")
        latest_transactions = predict_suspicious_transactions(latest_transactions, df)
        
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
            store_transaction(transaction_dict)
            
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

        connection = get_pg_connection()
        if connection is None:
            return jsonify({'error': 'Database connection failed.'}), 500

        cursor = connection.cursor()
        query = '''
            SELECT customer_id, name, age, annual_income, city_state, email
            FROM customers WHERE customer_id = %s
        '''
        cursor.execute(query, (customer_id,))
        result = cursor.fetchone()
        
        if result is None:
            cursor.close()
            connection.close()
            return jsonify({'error': 'Customer not found'}), 404
            
        customer_data = {
            'customer_id': result[0], 'name': result[1], 'age': result[2],
            'annual_income': float(result[3]) if result[3] else 0,
            'city_state': result[4], 'email': result[5]
        }
        
        cursor.close()
        connection.close()
        return jsonify(customer_data)
        
    except Exception as e:
        print(f"Error searching customer: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route("/calculate-risk-score", methods=['POST'])
def calculate_risk_score():
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400
            
        connection = get_pg_connection()
        if not connection:
            return jsonify({'error': 'Failed to connect to database'}), 500
            
        features = calculate_features(customer_id, connection)
        if not features:
            connection.close()
            return jsonify({'error': 'Failed to calculate features'}), 500
            
        feature_order = EXPECTED_FEATURES
        features_df = pd.DataFrame([features])[feature_order]
        
        print("\nFeatures for risk prediction:")
        for col in feature_order:
            print(f"{col}: {features_df[col].iloc[0]}")
        
        if risk_model is None:
            connection.close()
            return jsonify({'error': 'Risk prediction model not available'}), 500
        
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        for col in features_df.columns:
            features_df[col] = features_df[col].fillna(features_df[col].median())
        
        for col in features_df.columns:
            if col == 'age':
                features_df[col] = features_df[col] / 100.0
            elif col == 'annual_income':
                features_df[col] = features_df[col] / 1000000.0
            elif 'amount' in col:
                features_df[col] = features_df[col] / 100000.0
            elif col in ['transaction_frequency', 'total_transactions']:
                features_df[col] = features_df[col] / 1000.0
            else:
                max_val = features_df[col].max()
                if max_val > 0:
                    features_df[col] = features_df[col] / max_val
        
        risk_score = risk_model.predict(features_df)[0]
        risk_proba = risk_model.predict_proba(features_df)[0]
        
        print(f"\nRisk prediction: {risk_score}")
        print(f"Risk probabilities: {risk_proba}")
        
        risk_mapping = {
            0: {'level': 'Low Risk', 'color': '#00C851', 'description': 'Normal transaction patterns'},
            1: {'level': 'Medium Risk', 'color': '#ffbb33', 'description': 'Some unusual patterns detected'},
            2: {'level': 'High Risk', 'color': '#ff4444', 'description': 'Significant anomalies detected'}
        }
        
        risk_info = risk_mapping.get(risk_score, {'level': 'Unknown', 'color': '#ffffff', 'description': 'Unable to determine risk'})
        
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
    return render_template('select_report.html')

@app.route("/generate-pdf")
def generate_pdf_report():
    result = generate_pdf()
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return result

@app.route("/generate-csv")
def generate_csv_report():
    result = generate_csv()
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return result

if __name__ == "__main__":
    app.run(debug=True)