import pandas as pd
import psycopg2
from psycopg2 import Error

# PostgreSQL configuration
PG_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'radhika@28',
    'database': 'aml'   
}

try:
    # Load CSV
    print("Loading customer data from CSV...")
    df = pd.read_csv("C:/Users/Radhika/Downloads/customers_updated.csv")
    print(f"Loaded {len(df)} customers")

    # Connect to PostgreSQL
    print("\nConnecting to PostgreSQL...")
    conn = psycopg2.connect(**PG_CONFIG)
    cursor = conn.cursor()

    # Create customers table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255),
            age INTEGER,
            annual_income DECIMAL(15,2),
            city_state VARCHAR(255),
            email VARCHAR(255)
        )
    """)
    conn.commit()
    print("Ensured customers table exists")

    # Insert data into PostgreSQL
    print("\nInserting customer data...")
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO customers (customer_id, name, age, annual_income, city_state, email) 
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (customer_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    age = EXCLUDED.age,
                    annual_income = EXCLUDED.annual_income,
                    city_state = EXCLUDED.city_state,
                    email = EXCLUDED.email
            """, (
                row["customer_id"], 
                row["name"], 
                row["age"], 
                row["annual_income"], 
                row["city_state"], 
                row["email"]
            ))
        except Exception as e:
            print(f"Error inserting customer {row['customer_id']}: {str(e)}")
            continue

    # Commit and close
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Successfully loaded customer data")

except Error as e:
    print(f"Database error: {str(e)}")
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
        print("Database connection closed")
