# 🎓 Silent & At-Risk Student Detection System

> **TPC AI Platform** — Identify disengaged students, predict placement outcomes, and deliver autonomous personalised interventions using XGBoost + KMeans + a ReAct-pattern Agent.

---

## 🗂️ Project Structure

```
├── data_processor.py   # Data pipeline, feature engineering, Risk_Label
├── model_trainer.py    # XGBoost + KMeans training, InterventionAgent (ReAct)
├── app.py              # Streamlit TPC Dashboard (Admin + Student views)
├── requirements.txt    # Python dependencies
└── artefacts/          # Auto-created on first run
    ├── xgb_model.pkl
    ├── kmeans_model.pkl
    ├── processor.pkl
    ├── enriched_df.parquet
    └── metrics.json
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add the real dataset
Download from Kaggle:  
https://www.kaggle.com/datasets/sehaj1104/student-placement-prediction-dataset-2026  
Place the CSV anywhere and pass its path in `data_processor.py`:
```python
proc = StudentDataProcessor(csv_path="path/to/student_placement.csv")
```
If no CSV is provided the system auto-generates 500 realistic synthetic students.

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🧠 Architecture

### Data Pipeline (`data_processor.py`)
| Step | Description |
|------|-------------|
| `load()` | CSV or synthetic fallback |
| `clean()` | Null imputation, column normalisation, clipping |
| `engineer()` | `engagement_score = 0.45×attendance + 0.55×coding_skill` |
| | `Risk_Label = 1` if CGPA ≥ 7.5 AND unplaced |
| `encode()` | LabelEncoder for branch, gender, internship, backlogs |
| `scale()` | MinMaxScaler on all feature columns |

### Predictive Engine (`model_trainer.py`)
- **XGBoost Classifier** — 300 estimators, depth 5, trained on 80/20 split  
- **K-Means (k=3)** — clusters scored by centroid composite (CGPA + coding + engagement)  
  - 🟢 `Placement Ready` — high composite  
  - 🟡 `Silent/At-Risk` — mid composite, hidden underperformers  
  - 🔴 `Unprepared` — low composite  

### Agentic Layer — `InterventionAgent`
Implements a **Reason-Act (ReAct)** loop:
1. **OBSERVE** — loads student record
2. **REASON** — evaluates 8 dimensions against severity thresholds (critical / low / good)
3. **ACT** — selects context-aware actions per dimension; values are injected into text dynamically

No hard-coded strings — every suggestion references the student's actual score.

---

## 🖥️ Dashboard Views

### 🏛️ Admin Dashboard
- KPI row: total students, placed count, at-risk count, model AUC
- Cluster donut + branch risk heatmap
- Skill score bar chart + CGPA vs Placement Probability scatter
- **Urgent Intervention List** — ranked by urgency score
- **Automated Nudge Button** — simulates bulk email dispatch

### 👤 Student Portal
- **Career Health Meter** — placement probability gauge
- Skill radar chart (6-axis)
- Cluster badge + Risk flag
- **AI Action Plan** — personalised per-dimension steps
- Peer comparison bar chart
- Download action plan as JSON
- Request counsellor meeting

---

## ☁️ Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select `app.py`
3. No secrets needed (models auto-train on first run)

### Render
1. New Web Service → connect repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 📊 Feature Columns Used for Modelling

| Feature | Description |
|---------|-------------|
| `cgpa` | Cumulative Grade Point Average |
| `attendance_percentage` | Class attendance % |
| `coding_skill_score` | Coding proficiency score |
| `communication_skill_score` | Verbal/written communication |
| `technical_skill_score` | Domain technical knowledge |
| `projects_count` | Number of projects completed |
| `certifications_count` | Industry certifications earned |
| `internship_experience` | Binary: has internship? |
| `backlogs` | Binary: has academic backlogs? |
| `engagement_score` | Engineered: weighted attendance + coding |
| `gender` | Encoded |
| `branch` | Engineering branch (CSE, ECE, …) |

---

## 🎨 UI Design
Semi-dark glassmorphism theme with:
- **Palette**: Deep navy base · pastel blue (#6eb5ff) · soft purple (#b4a4f5) · teal (#5de0cb)
- **Fonts**: Syne (display/headers) + DM Sans (body)
- **Effects**: Frosted glass cards · subtle shadows · smooth gradient transitions
- **Charts**: Transparent Plotly backgrounds to blend with dark theme
