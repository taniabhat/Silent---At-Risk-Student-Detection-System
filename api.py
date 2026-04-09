import os
from sqlalchemy import create_engine
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from model_trainer import load_artefacts, InterventionAgent

ml_engine = {}

# DB Config
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///students.db")
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

db_engine = create_engine(DB_URL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Booting Microservice & Loading AI Models...")
    # SQLite local check
    if "sqlite" in DB_URL and not os.path.exists("students.db"):
        raise Exception("Database missing! Run 'py migrate.py' first.")
    
    art = load_artefacts()
    ml_engine["agent"] = InterventionAgent(raw_df=art["df"])
    print("✅ System Ready.")
    yield
    print("🛑 Shutting down.")

app = FastAPI(title="Silent Radar API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "Backend Active", "db": "PostgreSQL" if "postgres" in DB_URL else "SQLite"}

@app.get("/metrics")
def get_metrics():
    with db_engine.connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM students").scalar()
        at_risk = conn.execute("SELECT COUNT(*) FROM students WHERE cluster_label='Silent/At-Risk'").scalar()
        safe = conn.execute("SELECT COUNT(*) FROM students WHERE placement_prob >= 0.7").scalar()
    return {"n_total": total, "n_at_risk": at_risk, "n_safe": safe}

@app.get("/students/at-risk")
def get_urgent_list(top_n: int = 500):
    query = """
    SELECT student_id, branch, cgpa, coding_skill_score, attendance_percentage, 
           placement_prob, "Risk_Label", cluster_label, engagement_score
    FROM students
    WHERE (cluster_label = 'Silent/At-Risk' OR "Risk_Label" = 1)
    """
    df = pd.read_sql(query, db_engine)
    df = df[(df["placement_prob"] < 0.65) | (df["Risk_Label"] == 1)]
    df["urgency_score"] = df["Risk_Label"] * 2 + (1 - df["placement_prob"])
    df = df.sort_values("urgency_score", ascending=False).head(top_n)
    return df.to_dict(orient="records")

@app.get("/students/placed")
def get_placed_list(limit: int = 500):
    query = "SELECT * FROM students WHERE placement_status = 'Placed' LIMIT 500"
    df = pd.read_sql(query, db_engine)
    return df.to_dict(orient="records")

@app.get("/students/{student_id}")
def get_student(student_id: str):
    df = pd.read_sql(f"SELECT * FROM students WHERE student_id = {student_id}", db_engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    res = df.iloc[0].to_dict()
    
    plan = ml_engine["agent"].generate_plan(student_id)
    res["action_plan"] = plan
    return res

@app.get("/students/{student_id}/counsel")
def generate_counsel_report(student_id: str):
    from groq import Groq
    df = pd.read_sql(f"SELECT * FROM students WHERE student_id = {student_id}", db_engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    record = df.iloc[0].to_dict()
    plan = ml_engine["agent"].generate_plan(student_id)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            with open(".streamlit/secrets.toml", "rb") as f:
                secrets = tomllib.load(f)
                api_key = secrets.get("GROQ_API_KEY", "")
        except: pass
    if not api_key:
        return {"insight": "Groq API key not configured"}

    client = Groq(api_key=api_key)
    issues_text = "\n".join([f"- {i['dimension']}: {i['severity']}" for i in plan.get("issues", [])]) or "None"
    
    prompt = f"""You are an expert placement counselor. Analyze this student:
ID: {record.get('student_id')}
Branch: {record.get('branch')}
CGPA: {record.get('cgpa')}
Issues: {issues_text}

Provide 3 actionable recommendations and a warm closing message."""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250,
        )
        return {"insight": res.choices[0].message.content}
    except Exception as e:
        return {"insight": str(e)}
