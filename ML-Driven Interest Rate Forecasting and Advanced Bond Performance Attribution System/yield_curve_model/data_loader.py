import pandas as pd
import numpy as np
from datetime import datetime
import oracledb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Oracle Connection Details
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_content = os.getenv('DB_DSN')

class DataLoader:
    def __init__(self):
        self.user = DB_USER
        self.password = DB_PASS
        self.dsn = DB_content

    def get_connection(self):
        """Establishes connection to Oracle DB."""
        try:
            conn = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            return conn
        except oracledb.Error as e:
            print(f"Error connecting to Oracle DB: {e}")
            raise

    def load_market_data(self, start_date=None, end_date=None):
        """
        Loads market data from Oracle DB 'MARKET_DATA' table.
        Columns expected: day, tenor_1y, tenor_3y, tenor_5y, tenor_10y, tenor_20y, tenor_30y, tenor_50y
        """
        conn = self.get_connection()
        
        # Build Query
        # Note: Oracle usually stores column names in UPPERCASE unless quoted.
        query = """
            SELECT day as "date", 
                   tenor_1y, tenor_3y, tenor_5y, tenor_10y, 
                   tenor_20y, tenor_30y, tenor_50y 
            FROM MARKET_DATA
        """
        
        conditions = []
        params = {}
        
        if start_date:
            conditions.append("day >= :start_date")
            params['start_date'] = pd.to_datetime(start_date)
            
        if end_date:
            conditions.append("day <= :end_date")
            params['end_date'] = pd.to_datetime(end_date)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY day ASC"
        
        try:            
            df = pd.read_sql(query, conn, params=params, index_col='date', parse_dates=['date'])
            # Oracle returns columns in UPPERCASE by default. Normalize to lowercase.
            df.columns = df.columns.str.lower()
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        finally:
            conn.close()

    def load_bond_prices(self, bond_type='10y', start_date=None, end_date=None):
        """
        Loads FAIR_VALUE from KTB_10Y or KTB_50Y tables.
        Returns DataFrame with 'date' and 'fair_value'.
        """
        conn = self.get_connection()
        table_name = "KTB_10Y" if bond_type == '10y' else "KTB_50Y"
        
        query = f'SELECT day as "date", fair_value FROM {table_name}'
        
        conditions = []
        params = {}
        
        if start_date:
            conditions.append("day >= :start_date")
            params['start_date'] = pd.to_datetime(start_date)
            
        if end_date:
            conditions.append("day <= :end_date")
            params['end_date'] = pd.to_datetime(end_date)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY day ASC"
        
        try:
            df = pd.read_sql(query, conn, params=params, index_col='date', parse_dates=['date'])
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            print(f"Error loading bond prices for {table_name}: {e}")
            raise
        finally:
            conn.close()

if __name__ == "__main__":
    # Test Connection
    loader = DataLoader()
    try:
        print("Connecting to Oracle DB...")
        df = loader.load_market_data(start_date='2017-01-01', end_date='2025-12-31')
        print("Data Loaded Successfully:")
        print(df.head())
        print(f"Total Rows Fetched: {len(df)}")
        
        print("\nLoading KTB_10Y Prices...")
        df_10y = loader.load_bond_prices('10y', start_date='2017-01-01', end_date='2025-12-31')
        print(df_10y.head())
        print(f"Total Rows Fetched: {len(df_10y)}")

        print("\nLoading KTB_50Y Prices...")
        df_50y = loader.load_bond_prices('50y', start_date='2017-01-01', end_date='2025-12-31')
        print(df_50y.head())
        print(f"Total Rows Fetched: {len(df_50y)}")
        
    except Exception as e:
        print(f"Failed: {e}")
