# features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_features(customer_id, pg_connection):
    """Calculate derived features for the ML model"""
    cursor = None
    try:
        print(f"\n=== Calculating features for customer {customer_id} ===")
        cursor = pg_connection.cursor()
        
        # Get customer data
        cursor.execute('''
            SELECT annual_income, age
            FROM customers 
            WHERE customer_id = %s
        ''', (customer_id,))
        customer_data = cursor.fetchone()
        
        if not customer_data:
            print(f"No customer found with ID: {customer_id}")
            return None
            
        annual_income, age = customer_data
        annual_income = float(annual_income) if annual_income else 0.0
        print(f"Found customer: income={annual_income}, age={age}")
        
        # Get transaction data
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
            print(f"No transactions found for customer {customer_id}")
            return None
            
        total_transactions, total_amount, avg_amount, max_amount, unique_types = transaction_stats
        total_amount = float(total_amount) if total_amount else 0.0
        avg_amount = float(avg_amount) if avg_amount else 0.0
        max_amount = float(max_amount) if max_amount else 0.0
        print(f"Found transactions: count={total_transactions}, total={total_amount}")
        
        # Get transaction dates
        cursor.execute('''
            SELECT MIN(transaction_date), MAX(transaction_date)
            FROM transactions 
            WHERE customer_id = %s
        ''', (customer_id,))
        date_result = cursor.fetchone()
        first_date, last_date = date_result if date_result else (None, None)
        
        days_since_first = 0
        if first_date and last_date:
            days_since_first = (last_date - first_date).days
        
        transaction_frequency = total_transactions / days_since_first if days_since_first > 0 else 0.0
        amount_to_income_ratio = total_amount / annual_income if annual_income > 0 else 0.0
        
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
        
        print("\nSuccessfully calculated all features:")
        for name, value in features.items():
            print(f"{name}: {value}")
        
        return features
        
    except Exception as e:
        print(f"Error in calculate_features: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None
    finally:
        if cursor:
            cursor.close()

def extract_features(transaction, df):
    """Extract relevant features for Isolation Forest model"""
    try:
        amount = float(str(transaction["amount"]).replace(',', ''))
        txn_time = pd.to_datetime(transaction["transaction_time"])
        customer_transactions = df[df['customer_id'] == transaction['customer_id']].copy()
        customer_transactions['transaction_time'] = pd.to_datetime(customer_transactions['transaction_time'])
        customer_transactions['amount'] = customer_transactions['amount'].apply(lambda x: float(str(x).replace(',', '')))
        
        historical_transactions = customer_transactions[customer_transactions['transaction_time'] < txn_time]
        
        if historical_transactions.empty:
            print(f"\nNo historical transactions found for customer {transaction['customer_id']}")
            return [
                amount, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        
        transaction_type_mapping = {
            'deposit': 0, 'payment': 1, 'transfer': 2, 'withdrawal': 3
        }
        transaction_type_encoded = transaction_type_mapping.get(transaction["transaction_type"].lower(), 0)
        
        transaction_description_mapping = {
            'ATM Withdrawal': 0, 'Bank Transfer': 1, 'Cash Deposit': 2, 'Cash Withdrawal': 3,
            'Check Deposit': 4, 'Online Purchase': 5, 'Online Transaction': 6, 'Online Transfer': 7,
            'Restaurant Bill': 8, 'Salary Deposit': 9, 'Utility Bill': 10
        }
        transaction_description_encoded = transaction_description_mapping.get(transaction["transaction_description"], 0)
        
        seven_days_ago = txn_time - pd.Timedelta(days=7)
        recent_transactions = historical_transactions[historical_transactions['transaction_time'] >= seven_days_ago]
        transaction_count_7d = len(recent_transactions)
        
        last_txn_time = historical_transactions['transaction_time'].max()
        days_since_last_txn = (txn_time - last_txn_time).days
        
        thirty_days_ago = txn_time - pd.Timedelta(days=30)
        recent_30d = historical_transactions[historical_transactions['transaction_time'] >= thirty_days_ago]
        recent_7d = historical_transactions[historical_transactions['transaction_time'] >= seven_days_ago]
        
        txn_std_30d = recent_30d['amount'].std() if not recent_30d.empty else 0
        txn_mean_7d = recent_7d['amount'].mean() if not recent_7d.empty else 0
        txn_mean_30d = recent_30d['amount'].mean() if not recent_30d.empty else 0
        txn_ratio = amount / txn_mean_30d if txn_mean_30d != 0 else 1
        
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
            amount, transaction_type_encoded, transaction_description_encoded,
            transaction_count_7d, days_since_last_txn, txn_std_30d,
            txn_mean_7d, txn_mean_30d, txn_ratio
        ]
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None