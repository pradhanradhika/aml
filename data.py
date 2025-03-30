import pandas as pd
import mysql.connector

# Load CSV
df = pd.read_csv("C:/Users/Radhika/Downloads/customers_updated.csv")

# Connect to MySQL
conn = mysql.connector.connect(host="localhost", user="root", password="Ra@238gs", database="aml")
cursor = conn.cursor()

# Insert data into MySQL
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO customers (customer_id, name, age, annual_income, city_state, email) 
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (row["customer_id"], row["name"], row["age"], row["annual_income"], row["city_state"], row["email"]))

# Commit and close
conn.commit()
cursor.close()
conn.close()
