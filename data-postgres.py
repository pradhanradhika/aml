import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# PostgreSQL database URL 
DATABASE_URL = "postgresql://postgres:radhika%4028@localhost/aml"

# Create a database connection
engine = create_engine(DATABASE_URL)
conn = engine.raw_connection()
cursor = conn.cursor()

# Load the CSV file
csv_file = "C:/Users/Radhika/Downloads/customers_updated.csv"
df = pd.read_csv(csv_file)

# Convert DataFrame to list of tuples
data_tuples = [tuple(row) for row in df.itertuples(index=False, name=None)]

# SQL query for batch insertion
insert_query = """
INSERT INTO customers (customer_id, name, age, annual_income, city_state, email)
VALUES %s
"""

# Insert all rows at once using psycopg2 `execute_values`
from psycopg2.extras import execute_values
execute_values(cursor, insert_query.replace("VALUES %s", "VALUES %s"), data_tuples)

# Commit and close
conn.commit()
cursor.close()
conn.close()

print("All rows inserted successfully!")
