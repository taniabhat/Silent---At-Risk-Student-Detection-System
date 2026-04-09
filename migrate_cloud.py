import os
import pandas as pd
from sqlalchemy import create_engine

def main():
    db_url = input("Paste your Supabase Postgres URL starting with 'postgresql://':\n> ").strip()
    if not db_url.startswith("postgresql://"):
        print("Error: Must start with postgresql://")
        return
        
    print("Loading local SQLite database...")
    local_engine = create_engine("sqlite:///students.db")
    df = pd.read_sql("SELECT * FROM students", local_engine)
    
    print(f"Connecting to Supabase... (Pushing {len(df)} rows)")
    cloud_engine = create_engine(db_url)
    df.to_sql("students", con=cloud_engine, if_exists="replace", index=False)
    print("✅ Successfully pushed all students to Supabase Cloud Database!")

if __name__ == "__main__":
    main()
