import pandas as pd
from sqlalchemy import create_engine
import os
import sys

def main():
    print("Starting Data Migration to Database...")
    
    # 1. Ensure artefacts exist
    if not os.path.exists("artefacts/enriched_df.parquet"):
        print("Artefacts not found. Run 'py model_trainer.py' first.")
        sys.exit(1)
        
    print("Loading processed data...")
    # Using enriched_df to retain original feature values
    df = pd.read_parquet("artefacts/enriched_df.parquet")
    
    # 2. Connect to local SQLite DB (Stand-in for Supabase)
    print("Connecting to SQLite Database...")
    engine = create_engine("sqlite:///students.db")
    
    # 3. Write chunked to SQL
    print(f"Dumping {len(df)} rows into 'students' table...")
    df.to_sql("students", con=engine, if_exists="replace", index=False)
    
    print("Migration Complete! You can now query students.db instantly for 0 cost.")

if __name__ == "__main__":
    main()
