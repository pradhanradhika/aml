# database.py
import psycopg2
from psycopg2 import Error
from config import PG_CONFIG 

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