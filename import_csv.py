import os
import sys
import argparse
import pandas as pd
import sqlalchemy
from sqlalchemy import text

def parse_args():
    parser = argparse.ArgumentParser(description="Import Google Play CSV data into MySQL staging tables.")
    parser.add_argument("--host", default="localhost", help="MySQL database host (default: localhost)")
    parser.add_argument("--port", type=int, default=3306, help="MySQL database port (default: 3306)")
    parser.add_argument("--user", default="root", help="MySQL database username (default: root)")
    parser.add_argument("--password", default="", help="MySQL database password (default: empty)")
    parser.add_argument("--database", default="Google_Play", help="MySQL database name (default: Google_Play)")
    parser.add_argument("--no-truncate", action="store_true", help="Do not truncate staging tables before importing")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Verify CSV files exist
    apps_csv = "data/googleplaystore.csv"
    reviews_csv = "data/googleplaystore_user_reviews.csv"
    
    for path in [apps_csv, reviews_csv]:
        if not os.path.exists(path):
            print(f"Error: Required CSV file not found at '{path}'", file=sys.stderr)
            sys.exit(1)

    # 2. Establish connection to MySQL database
    # Use URL.create to safely escape special characters in the password
    from sqlalchemy.engine import URL
    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        database=args.database
    )
    
    print(f"Connecting to MySQL database '{args.database}' on {args.host}:{args.port} as user '{args.user}'...")
    try:
        engine = sqlalchemy.create_engine(connection_url)
        # Test connection
        with engine.connect() as conn:
            pass
        print("Database connection established successfully.")
    except Exception as e:
        print("\nDatabase connection failed!", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        print("\nHow to troubleshoot:", file=sys.stderr)
        print("1. Ensure your MySQL server is running.", file=sys.stderr)
        print("2. Check if the database 'Google_Play' exists.", file=sys.stderr)
        print("3. Check username, password, host and port.", file=sys.stderr)
        print("   Example run with custom credentials:", file=sys.stderr)
        print(f"   python import_csv.py --user your_user --password your_password --port 3306", file=sys.stderr)
        sys.exit(1)

    # 3. Truncate tables if requested
    if not args.no_truncate:
        print("\nTruncating staging tables...")
        try:
            with engine.begin() as conn:
                conn.execute(text("TRUNCATE TABLE stg_google_play_apps;"))
                conn.execute(text("TRUNCATE TABLE stg_google_play_reviews;"))
            print("Staging tables truncated successfully.")
        except Exception as e:
            print(f"Warning: Failed to truncate staging tables: {e}", file=sys.stderr)
            print("Proceeding with append import...", file=sys.stderr)

    # 4. Load & Import googleplaystore.csv (apps)
    print("\nReading googleplaystore.csv...")
    try:
        df_apps = pd.read_csv(apps_csv, dtype=str)
        # Normalize column names: lowercase and replace space with underscore
        df_apps.columns = df_apps.columns.str.lower().str.replace(' ', '_')
        # Map NaNs / empty rows to None so they insert as SQL NULL
        df_apps = df_apps.astype(object).where(pd.notnull(df_apps), None)
        
        apps_row_count = len(df_apps)
        print(f"Parsed {apps_row_count} rows from CSV. Importing into 'stg_google_play_apps'...")
        
        # Load into MySQL
        df_apps.to_sql('stg_google_play_apps', engine, if_exists='append', index=False)
        print(f"Successfully imported {apps_row_count} rows into 'stg_google_play_apps'.")
    except Exception as e:
        print(f"Error importing apps: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Load & Import googleplaystore_user_reviews.csv (reviews)
    print("\nReading googleplaystore_user_reviews.csv...")
    try:
        df_reviews = pd.read_csv(reviews_csv, dtype=str)
        # Normalize column names: lowercase and replace space with underscore
        df_reviews.columns = df_reviews.columns.str.lower().str.replace(' ', '_')
        # Map NaNs / empty rows to None so they insert as SQL NULL
        df_reviews = df_reviews.astype(object).where(pd.notnull(df_reviews), None)
        
        reviews_row_count = len(df_reviews)
        print(f"Parsed {reviews_row_count} rows from CSV. Importing into 'stg_google_play_reviews'...")
        
        # Load into MySQL
        df_reviews.to_sql('stg_google_play_reviews', engine, if_exists='append', index=False)
        print(f"Successfully imported {reviews_row_count} rows into 'stg_google_play_reviews'.")
    except Exception as e:
        print(f"Error importing reviews: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "="*40)
    print("IMPORT COMPLETE SUCCESSFULY")
    print(f"Staging Apps:    {apps_row_count} rows")
    print(f"Staging Reviews: {reviews_row_count} rows")
    print("="*40)

if __name__ == "__main__":
    main()
