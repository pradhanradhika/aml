CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    customer_id VARCHAR(255) NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    transaction_amount DECIMAL(15,2) NOT NULL,
    merchant_name VARCHAR(255),
    merchant_category VARCHAR(255),
    transaction_type VARCHAR(50),
    risk_score DECIMAL(5,2),
    is_suspicious BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitored_transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    customer_id VARCHAR(255) NOT NULL,
    transaction_time TIMESTAMP NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    transaction_type VARCHAR(50),
    transaction_description VARCHAR(255),
    is_suspicious BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
