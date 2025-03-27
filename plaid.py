from flask import Flask
from models import db, Transaction
from plaid import Configuration, ApiClient
from plaid.api.plaid_api import PlaidApi  # Updated import
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.country_code import CountryCode
from datetime import datetime, timedelta

app = Flask(__name__)

# ✅ Database Configuration
DATABASE_URL = "postgresql://postgres:radhika%4028@localhost/aml"
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# ✅ Plaid API Credentials
PLAID_CLIENT_ID = "67d1adf7859c7e0025980e09"
PLAID_SECRET = "20a92ad8f600d08cc8e17ba94dc75d"
PLAID_ENV = "sandbox"  # Change to 'development' or 'production' if needed
ACCESS_TOKEN = "access-sandbox-0cf4ea2f-6503-4c1f-b98f-282285628da8"

# ✅ Plaid API Client Setup
config = Configuration(
    host="https://sandbox.plaid.com",
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET
    }
)
api_client = ApiClient(configuration=config)
client = PlaidApi(api_client)  # Updated client initialization

# ✅ Function to Fetch Transactions
def fetch_plaid_transactions():
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    request = TransactionsGetRequest(
        access_token=ACCESS_TOKEN,
        start_date=start_date,
        end_date=end_date,
        options=TransactionsGetRequestOptions(count=100)
    )

    try:
        response = client.transactions_get(request)  # Updated method call
        return response["transactions"]
    except Exception as e:
        print(f"❌ Plaid API error: {e}")
        return []

# ✅ Store Transactions in Database
def store_transactions():
    transactions = fetch_plaid_transactions()
    for txn in transactions:
        new_transaction = Transaction(
            account_id=txn.account_id,
            transaction_id=txn.transaction_id,
            amount=txn.amount,
            currency=txn.iso_currency_code,
            category=txn.category[0] if txn.category else "Unknown",
            merchant_name=txn.merchant_name if txn.merchant_name else "Unknown",
            date=txn.date
        )
        db.session.add(new_transaction)

    db.session.commit()
    print("✅ Transactions stored successfully!")

# ✅ Route to Manually Trigger Transaction Updates
@app.route("/update-transactions")
def update_transactions():
    store_transactions()
    return "✅ Transactions updated successfully!"

@app.route("/")
def home():
    return "Flask app is running with Plaid API!"

if __name__ == "__main__":
    app.run(debug=True)

    