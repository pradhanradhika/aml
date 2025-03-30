
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# PostgreSQL database URL (Replace with your actual credentials)
DATABASE_URL = "postgresql://postgres:radhika%4028@localhost/aml"

# Establish a connection
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Load the CSV file
csv_file = "C:/Users/Radhika/Downloads/transactions_updated.csv"
df = pd.read_csv(csv_file)

# Ensure column names match exactly with your PostgreSQL table
df.columns = ["transaction_id", "customer_id", "transaction_date", "transaction_amount", "transaction_type", "transaction_description"]

# Convert transaction_date to datetime format (PostgreSQL requires proper date format)
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# Convert DataFrame to list of tuples
data_tuples = [tuple(row) for row in df.itertuples(index=False, name=None)]

# SQL query for batch insertion
insert_query = """
INSERT INTO transactions (transaction_id, customer_id, transaction_date, transaction_amount, transaction_type, transaction_description)
VALUES %s
"""

# Insert all rows at once using execute_values
execute_values(cursor, insert_query, data_tuples)

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

print("All transaction rows inserted successfully!")
