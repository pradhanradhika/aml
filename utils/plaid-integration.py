from flask import Flask, request, jsonify
from datetime import datetime, timedelta, date
import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# Initialize Flask app
app = Flask(__name__)

# Plaid API configuration
PLAID_CLIENT_ID = "67d1adf7859c7e0025980e09"
PLAID_SECRET = "20a92ad8f600d08cc8e17ba94dc75d"
PLAID_ACCESS_TOKEN = "access-sandbox-0cf4ea2f-6503-4c1f-b98f-282285628da8"

configuration = plaid.Configuration(
    host=plaid.Environment.Sandbox,
    api_key={
        'clientId': PLAID_CLIENT_ID,
        'secret': PLAID_SECRET,
    }
)
api_client = plaid.ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

# PostgreSQL database setup
DATABASE_URI = 'postgresql://postgres:radhika%4028@localhost:5432/aml'
engine = create_engine(DATABASE_URI)
Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, unique=True)
    account_id = Column(String, nullable=False)  # Ensure account_id is required
    amount = Column(Float)
    currency = Column(String, nullable=False)  # Add currency
    date = Column(DateTime)
    merchant_name = Column(String)
    category = Column(String)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Function to fetch transactions from Plaid
def get_transactions(access_token, start_date, end_date):
    print(f"Making Plaid API request with start_date={start_date}, end_date={end_date}")
    request = TransactionsGetRequest(
        access_token=access_token,
        start_date=start_date,
        end_date=end_date,
        options=TransactionsGetRequestOptions()
    )
    response = client.transactions_get(request)
    print(f"Fetched {len(response['transactions'])} transactions")  # Log the number of transactions
    return response['transactions']

# Root route
@app.route('/')
def home():
    return "Welcome to the Plaid Integration API!"

# Webhook endpoint for real-time updates
@app.route('/plaid_webhook', methods=['POST'])
def plaid_webhook():
    data = request.json
    if data['webhook_type'] == 'TRANSACTIONS':
        if data['webhook_code'] == 'DEFAULT_UPDATE':
            # Fetch new transactions
            start_date = (datetime.now() - timedelta(days=30)).date()
            end_date = datetime.now().date()
            new_transactions = get_transactions(PLAID_ACCESS_TOKEN, start_date, end_date)

            # Save new transactions to the database
            for transaction in new_transactions:
                try:
                    if not session.query(Transaction).filter_by(transaction_id=transaction['transaction_id']).first():
                        new_transaction = Transaction(
                            transaction_id=transaction['transaction_id'],
                            account_id=transaction['account_id'],
                            amount=transaction['amount'],
                            currency=transaction['iso_currency_code'],  # Add currency
                            date=transaction['date'],
                            merchant_name=transaction['merchant_name'],
                            category=','.join(transaction['category']),
                        )
                        session.add(new_transaction)
                        session.commit()
                except Exception as e:
                    session.rollback()  # Roll back the session on error
                    print(f"Error saving transaction: {e}")
    return jsonify({'status': 'success'}), 200

# Endpoint to manually fetch and store transactions (for testing)
@app.route('/fetch_transactions', methods=['GET'])
def fetch_transactions():
    print("Fetching transactions...")
    start_date = (datetime.now() - timedelta(days=30)).date()
    end_date = datetime.now().date()
    transactions = get_transactions(PLAID_ACCESS_TOKEN, start_date, end_date)

    for transaction in transactions:
        try:
            if not session.query(Transaction).filter_by(transaction_id=transaction['transaction_id']).first():
                new_transaction = Transaction(
                    transaction_id=transaction['transaction_id'],
                    account_id=transaction['account_id'],
                    amount=transaction['amount'],
                    currency=transaction['iso_currency_code'],  # Add currency
                    date=transaction['date'],
                    merchant_name=transaction['merchant_name'],
                    category=','.join(transaction['category']),
                )
                session.add(new_transaction)
                session.commit()
        except Exception as e:
            session.rollback()  # Roll back the session on error
            print(f"Error saving transaction: {e}")
    return jsonify({'status': 'transactions fetched and stored'}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000, debug=True)