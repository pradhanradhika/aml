from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Transaction(db.Model):
    __tablename__ = "transactions"

    id = db.Column(db.Integer, primary_key=True)  # Unique ID for each transaction
    account_id = db.Column(db.String(100), nullable=False)  # Plaid account ID
    transaction_id = db.Column(db.String(100), unique=True, nullable=False)  # Plaid transaction ID
    amount = db.Column(db.Float, nullable=False)  # Transaction amount
    currency = db.Column(db.String(10), nullable=False)  # Currency type (USD, INR, etc.)
    category = db.Column(db.String(200))  # Transaction category
    merchant_name = db.Column(db.String(200))  # Merchant name (if available)
    date = db.Column(db.Date, nullable=False)  # Transaction date
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp when added to DB
