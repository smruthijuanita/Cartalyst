import pandas as pd
import sqlite3
import logging

# --- Configuration ---
CSV_FILE_PATH = 'cleaned_data.csv'
DB_FILE_PATH = 'parts.db'
TABLE_NAME = 'parts'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        conn.close()
        
        logger.info(f"✅ Success! Database '{DB_FILE_PATH}' created with {len(df)} records in table '{TABLE_NAME}'.")

    except FileNotFoundError:
        logger.error(f"❌ ERROR: The source file '{CSV_FILE_PATH}' was not found.")
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")

if __name__ == '__main__':
    create_database()