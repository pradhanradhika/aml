import psycopg2
from psycopg2 import Error

# PostgreSQL configuration
PG_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'radhika@28',
    'database': 'aml'   
}

def reset_database():
    """Reset the database tables"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**PG_CONFIG)
        cur = conn.cursor()
        
        # Drop existing tables
        cur.execute('DROP TABLE IF EXISTS monitored_transactions;')
        cur.execute('DROP TABLE IF EXISTS transactions;')
        conn.commit()
        print("✅ Dropped existing tables")
        
        # Read schema.sql
        with open('schema.sql', 'r') as f:
            schema_sql = f.read()
        
        # Create tables
        cur.execute(schema_sql)
        conn.commit()
        print("✅ Created new tables")
        
        cur.close()
        conn.close()
        print("✅ Database reset complete")
        
    except Error as e:
        print(f"Error resetting database: {e}")

if __name__ == "__main__":
    reset_database()
