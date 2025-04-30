# transactions.py
import pandas as pd
from database import get_pg_connection
from config import DATA_PATH

def load_data():
    """Load the dataset"""
    try:
        print("\nLoading transaction data from CSV...")
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
        
        # Clean column names
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
        
        # Drop invalid rows
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
    df = load_data()
    
    if df.empty:
        print("No data loaded from CSV")
        return pd.DataFrame(), df
    
    print(f"Loaded {len(df)} total transactions")
    
    # Get processed transaction IDs
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
    
    # Convert transaction_id to string
    df['transaction_id'] = df['transaction_id'].astype(str)
    
    # Filter unprocessed transactions
    unprocessed_df = df[~df['transaction_id'].isin(processed_ids)]
    print(f"Found {len(unprocessed_df)} unprocessed transactions")
    
    if len(unprocessed_df) == 0:
        print("No new transactions to process")
        return pd.DataFrame(), df
    
    # Get 10 unprocessed transactions
    latest_transactions = unprocessed_df.head(10).copy()
    print(f"Selected {len(latest_transactions)} new transactions to display")
    
    return latest_transactions, df