# config.py
PG_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'radhika@28',
    'database': 'aml'
}

MODEL_PATHS = {
    'random_forest': 'random_forest_model.pkl',
    'isolation_forest': 'isoforest_model.pkl'
}

# Random forest features
EXPECTED_FEATURES = [
    'annual_income',
    'total_transactions',
    'total_amount_spent',
    'average_transaction_amount',
    'max_transaction_amount',
    'unique_transaction_descriptions',
    'days_since_first_transaction',
    'transaction_frequency',
    'transaction_amount_to_income_ratio',
    'age'
]

DATA_PATH = "C:/Users/Radhika/.vscode/AML/bank_transactions_trimmed.csv"