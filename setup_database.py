import pandas as pd
import sqlite3
import logging
from datetime import datetime

# --- Configuration ---
CSV_FILE_PATH = 'cleaned_parts.csv'
DB_FILE_PATH = 'parts.db'
TABLE_NAME = 'parts'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_cart_and_orders_tables(conn):
    """Create cart and orders tables for shopping functionality."""
    cursor = conn.cursor()
    
    # Create users table for better user management
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create cart table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            part_no TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, part_no)
        )
    ''')
    
    # Create orders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_amount REAL NOT NULL,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create order_items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            part_no TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            total_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    ''')
    
    # Insert default users
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, role) VALUES
        ('customer', 'customer'),
        ('employee', 'employee')
    ''')
    
    conn.commit()
    logger.info("✅ Cart and orders tables created successfully.")

def create_database():
    """Reads data from the CSV, cleans it, and saves it to a SQLite database."""
    logger.info(f"Starting database setup from '{CSV_FILE_PATH}'...")
    
    try:
        # Define column names since the CSV has no header
        column_names = [
            "TransactionID", "Date", "CustomerID", "PartNo", "Quantity",
            "Rate", "TotalPrice", "PartDescription", "Category", "Source", "VehicleMake"
        ]
        df = pd.read_csv(CSV_FILE_PATH, header=None, names=column_names, on_bad_lines='skip')
        
        # --- Basic Data Cleaning ---
        df = df.dropna(subset=['PartNo', 'PartDescription'])
        df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df = df.dropna(subset=['Rate', 'Quantity'])
        df['Quantity'] = df['Quantity'].astype(int)

        # Add a unique ID column, which is good practice for databases
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)

        # Connect to SQLite and save the data
        conn = sqlite3.connect(DB_FILE_PATH)
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        
        # Create additional tables for cart and orders
        create_cart_and_orders_tables(conn)
        
        conn.close()
        
        logger.info(f"✅ Success! Database '{DB_FILE_PATH}' created with {len(df)} records in table '{TABLE_NAME}'.")
        logger.info("✅ Additional tables for cart and orders functionality created.")

    except FileNotFoundError:
        logger.error(f"❌ ERROR: The source file '{CSV_FILE_PATH}' was not found.")
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")

if __name__ == '__main__':
    create_database()