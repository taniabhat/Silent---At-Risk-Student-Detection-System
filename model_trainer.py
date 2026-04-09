"""
model_trainer.py
----------------
Silent & At-Risk Student Detection System
Predictive Engine + Agentic Intervention Layer

Exports
-------
  train_and_save()     – full training pipeline, persists artefacts to disk
  load_artefacts()     – reload saved models + processor
  InterventionAgent    – ReAct-pattern agent for personalised action plans
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ARTEFACT_DIR   = "artefacts"
XGB_PATH       = os.path.join(ARTEFACT_DIR, "xgb_model.pkl")
KMEANS_PATH    = os.path.join(ARTEFACT_DIR, "kmeans_model.pkl")
PROCESSOR_PATH = os.path.join(ARTEFACT_DIR, "processor.pkl")
METRICS_PATH   = os.path.join(ARTEFACT_DIR, "metrics.json")

# ─────────────────────────────────────────────
# CLUSTER LABELS  (assigned post-fit via centroid analysis)
# ─────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "Silent/At-Risk",
    1: "Placement Ready",
    2: "Unprepared",
}


# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_and_save(csv_path: str | None = None) -> dict:
    """
    Full pipeline: process data → train XGB → train KMeans →
    assign cluster labels → save artefacts → return metrics.
    """
    # Late import to avoid circular dependency
    from data_processor import StudentDataProcessor, FEATURE_COLS, DEFAULT_CSV

    os.makedirs(ARTEFACT_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────
    # Use the bundled real dataset by default
    proc = StudentDataProcessor(csv_path=csv_path or DEFAULT_CSV)
    proc.run()
    X, y = proc.get_features()
    df   = proc.get_unscaled_dataframe()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── XGBoost Classifier ───────────────────
    xgb = XGBClassifier(
        n_estimators    = 300,
        max_depth       = 5,
        learning_rate   = 0.07,
        subsample       = 0.85,
        colsample_bytree= 0.85,
        use_label_encoder=False,
        eval_metric     = "logloss",
        random_state    = 42,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred     = xgb.predict(X_test)
    y_prob     = xgb.predict_proba(X_test)[:, 1]
    accuracy   = accuracy_score(y_test, y_pred)
    roc_auc    = roc_auc_score(y_test, y_prob)
    report     = classification_report(y_test, y_pred, output_dict=True)

    # ── K-Means Clustering ───────────────────
    # Use placement probability + key skill columns as clustering features
    feat_present = [c for c in FEATURE_COLS if c in X.columns]
    cluster_feats = [
        c for c in feat_present
        if c in ["cgpa", "coding_skill_score", "engagement_score",
                 "technical_skill_score", "attendance_percentage",
                 "communication_skill_score"]
    ]
    # Subset X with only clustering features (already scaled)
    X_cluster = X[cluster_feats] if cluster_feats else X

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=15, max_iter=500)
    raw_clusters = kmeans.fit_predict(X_cluster)

    # ── Assign Semantic Labels via Centroid Ranking ──
    # Score each centroid: high cgpa + coding → "Placement Ready"
    #                      mid cgpa + low engagement → "Silent/At-Risk"
    #                      low everything → "Unprepared"
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_feats)
    # Composite centroid score
    score_cols = [c for c in ["cgpa", "coding_skill_score", "engagement_score"] if c in centroids.columns]
    if score_cols:
        centroids["composite"] = centroids[score_cols].mean(axis=1)
        rank_order = centroids["composite"].rank(ascending=False).astype(int)
        # rank 1 = highest → Placement Ready, rank 2 → Silent/At-Risk, rank 3 → Unprepared
        cluster_label_map = {}
        for cid, rk in rank_order.items():
            if rk == 1:
                cluster_label_map[cid] = "Placement Ready"
            elif rk == 2:
                cluster_label_map[cid] = "Silent/At-Risk"
            else:
                cluster_label_map[cid] = "Unprepared"
    else:
        cluster_label_map = {i: list(CLUSTER_NAMES.values())[i] for i in range(3)}

    # Attach to df
    df["raw_cluster"]    = raw_clusters
    df["cluster_label"]  = df["raw_cluster"].map(cluster_label_map)
    df["placement_prob"] = xgb.predict_proba(X)[:, 1]

    # ── Metrics ──────────────────────────────
    metrics = {
        "accuracy":        round(accuracy, 4),
        "roc_auc":         round(roc_auc, 4),
        "n_samples":       len(df),
        "n_placed":        int(df["placed"].sum()),
        "cluster_counts":  df["cluster_label"].value_counts().to_dict(),
        "risk_count":      int(df["Risk_Label"].sum()),
        "classification_report": report,
        "cluster_label_map":     cluster_label_map,
        "cluster_feats":         cluster_feats,
    }

    # ── Save ─────────────────────────────────
    with open(XGB_PATH, "wb")       as f: pickle.dump(xgb,    f)
    with open(KMEANS_PATH, "wb")    as f: pickle.dump(kmeans,  f)
    with open(PROCESSOR_PATH, "wb") as f: pickle.dump(proc,    f)
    with open(METRICS_PATH, "w")    as f: json.dump(metrics,   f, indent=2)

    # Save enriched df
    df.to_parquet(os.path.join(ARTEFACT_DIR, "enriched_df.parquet"), index=False)

    print(f"[Trainer] Accuracy: {accuracy:.4f}  ROC-AUC: {roc_auc:.4f}")
    print(f"[Trainer] Cluster distribution: {metrics['cluster_counts']}")
    print(f"[Trainer] Artefacts saved to '{ARTEFACT_DIR}/'")
    return metrics


# ─────────────────────────────────────────────
# ARTEFACT LOADER
# ─────────────────────────────────────────────
def load_artefacts() -> dict:
    """
    Load persisted models and data from disk.
    Returns dict with keys: xgb, kmeans, processor, metrics, df
    """
    artefacts = {}
    missing = []
    for name, path in [("xgb", XGB_PATH), ("kmeans", KMEANS_PATH),
                       ("processor", PROCESSOR_PATH)]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                artefacts[name] = pickle.load(f)
        else:
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            f"[Trainer] Artefacts not found: {missing}. Run train_and_save() first."
        )

    with open(METRICS_PATH) as f:
        artefacts["metrics"] = json.load(f)

    df_path = os.path.join(ARTEFACT_DIR, "enriched_df.parquet")
    if os.path.exists(df_path):
        artefacts["df"] = pd.read_parquet(df_path)
    else:
        artefacts["df"] = None

    return artefacts


# ─────────────────────────────────────────────
# REACT-PATTERN INTERVENTION AGENT
# ─────────────────────────────────────────────
class InterventionAgent:
    """
    Autonomous Intervention Agent using a Reason-Act (ReAct) pattern.

    For each at-risk student, the agent:
      1. OBSERVES  – gathers student's feature vector
      2. REASONS   – identifies the weakest dimensions
      3. ACTS      – generates a prioritised, personalised Action Plan

    All suggestions are context-aware and dynamically generated;
    no hard-coded string is ever returned without examining the data.
    """

    # ── Threshold maps for each dimension ────
    THRESHOLDS = {
        "coding_skill_score":        {"critical": 40, "low": 60, "good": 75},
        "communication_skill_score": {"critical": 40, "low": 60, "good": 75},
        "technical_skill_score":     {"critical": 40, "low": 60, "good": 75},
        "attendance_percentage":     {"critical": 55, "low": 70, "good": 85},
        "cgpa":                      {"critical": 5.0, "low": 6.5, "good": 7.5},
        "projects_count":            {"critical": 0,  "low": 1,   "good": 3},
        "certifications_count":      {"critical": 0,  "low": 1,   "good": 3},
        "engagement_score":          {"critical": 40, "low": 55,  "good": 70},
    }

    # ── Action templates per dimension × severity ─
    ACTION_TEMPLATES = {
        "coding_skill_score": {
            "critical": [
                "🔴 URGENT: Start with LeetCode Easy – focus on Arrays & Strings daily (1 hr/day)",
                "📘 Complete 'Python for Data Structures' on Coursera (free audit)",
                "🤝 Join your college Coding Club immediately for peer support",
            ],
            "low": [
                "🟡 Solve 3 LeetCode Medium problems per week (Hash Maps, Trees)",
                "📗 Practice system design basics on 'neetcode.io'",
                "🎯 Attempt a HackerRank 30-day coding challenge",
            ],
            "good": [
                "🟢 Push to LeetCode Hard – focus on Dynamic Programming",
                "🏆 Register for an upcoming competitive programming contest",
            ],
        },
        "communication_skill_score": {
            "critical": [
                "🔴 URGENT: Enroll in Toastmasters or college communication workshop",
                "🎤 Practice mock HR interviews with a peer 3×/week",
                "📹 Record yourself answering 'Tell me about yourself' and review",
            ],
            "low": [
                "🟡 Practice STAR-method answers for top 20 interview questions",
                "📖 Read 'Cracking the Coding Interview' – soft-skills chapter",
                "🎙️ Attend at least 2 webinars and ask questions publicly",
            ],
            "good": [
                "🟢 Lead a team presentation or technical talk in class",
            ],
        },
        "technical_skill_score": {
            "critical": [
                "🔴 URGENT: Revise core CS fundamentals – OS, DBMS, Networks",
                "📚 Complete GATE-level MCQs on subject fundamentals daily",
                "🖥️ Build one mini-project using your strongest language this week",
            ],
            "low": [
                "🟡 Pick one domain (Web/ML/Cloud) and complete a specialisation",
                "🧪 Work through 2 hands-on labs on AWS Free Tier or Google Colab",
            ],
            "good": [
                "🟢 Contribute to an open-source project on GitHub",
                "☁️ Prepare for a cloud certification (AWS/Azure fundamentals)",
            ],
        },
        "attendance_percentage": {
            "critical": [
                "🔴 URGENT: Attendance is critically low – contact your academic advisor today",
                "📅 Set daily phone reminders for every class",
                "🧑‍🤝‍🧑 Find a study buddy to keep each other accountable",
            ],
            "low": [
                "🟡 Target 85%+ attendance this month – track it in a habit app",
                "📓 Review all missed lecture notes before the next class",
            ],
            "good": [
                "🟢 Maintain consistency and mentor peers who struggle with attendance",
            ],
        },
        "cgpa": {
            "critical": [
                "🔴 URGENT: Identify your 2 weakest subjects and seek tutoring this week",
                "📝 Create a semester study plan with weekly milestones",
                "👨‍🏫 Visit your professors during office hours – explain your situation",
            ],
            "low": [
                "🟡 Aim for a 0.3 CGPA improvement this semester with focused revision",
                "🗂️ Use the Pomodoro technique for exam preparation",
            ],
            "good": [
                "🟢 Apply for research internships or TA positions",
            ],
        },
        "projects_count": {
            "critical": [
                "🔴 URGENT: Start a 2-week mini-project TODAY (a simple CRUD app counts!)",
                "🌐 Deploy your project on GitHub and add a README",
            ],
            "low": [
                "🟡 Add 1 new project per month – use a public dataset from Kaggle",
                "🤝 Collaborate on a hackathon project this semester",
            ],
            "good": [
                "🟢 Aim for a publication-quality project or a startup prototype",
            ],
        },
        "certifications_count": {
            "critical": [
                "🔴 URGENT: Complete a free Google or IBM certification on Coursera",
                "📜 Certifications signal commitment to recruiters – get at least 1",
            ],
            "low": [
                "🟡 Earn 1 industry certification (AWS/Google/Microsoft) this semester",
            ],
            "good": [
                "🟢 Stack specialised certs (e.g., TensorFlow Developer, Azure AI)",
            ],
        },
        "engagement_score": {
            "critical": [
                "🔴 URGENT: Overall engagement is very low – meet with your placement coordinator",
                "🗓️ Schedule at least 2 hrs/day dedicated to placement preparation",
            ],
            "low": [
                "🟡 Join study groups and placement prep communities online",
            ],
            "good": [
                "🟢 Help organise a college placement bootcamp – leadership visibility",
            ],
        },
    }

    def __init__(self, raw_df: pd.DataFrame):
        """
        raw_df: the *un-scaled* dataframe (with original numeric values)
                so that thresholds are meaningful.
        """
        self.raw_df = raw_df.copy()

    # ── OBSERVE ──────────────────────────────
    def _observe(self, student_id: str) -> dict:
        row = self.raw_df[self.raw_df["student_id"] == student_id]
        if row.empty:
            raise ValueError(f"Student '{student_id}' not found.")
        return row.iloc[0].to_dict()

    # ── REASON ───────────────────────────────
    def _reason(self, record: dict) -> list[dict]:
        """
        Evaluate each tracked dimension and return a list of
        weaknesses sorted by severity (critical first).
        """
        issues = []
        for dim, thresholds in self.THRESHOLDS.items():
            val = record.get(dim)
            if val is None:
                continue
            if val < thresholds["critical"]:
                severity = "critical"
                priority = 1
            elif val < thresholds["low"]:
                severity = "low"
                priority = 2
            elif val < thresholds["good"]:
                severity = "good"
                priority = 3
            else:
                continue   # performing well – no action needed

            issues.append({
                "dimension": dim,
                "value":     round(float(val), 2),
                "severity":  severity,
                "priority":  priority,
                "threshold": thresholds,
            })

        issues.sort(key=lambda x: x["priority"])
        return issues

    # ── ACT ──────────────────────────────────
    def _act(self, issues: list[dict]) -> list[dict]:
        """
        For each identified issue, select context-aware actions.
        Returns a list of action-plan items.
        """
        plan = []
        for issue in issues[:5]:   # cap at top-5 priorities
            dim      = issue["dimension"]
            severity = issue["severity"]
            actions  = self.ACTION_TEMPLATES.get(dim, {}).get(severity, [])

            # Dynamically add context to action text
            contextualised = []
            for action in actions:
                if dim == "coding_skill_score" and severity == "critical":
                    action = action.replace("daily (1 hr/day)",
                                            f"daily (current score: {issue['value']:.0f}/100)")
                elif dim == "attendance_percentage":
                    action = action.replace("critically low",
                                            f"critically low at {issue['value']:.1f}%")
                contextualised.append(action)

            plan.append({
                "dimension":    dim.replace("_", " ").title(),
                "current_val":  issue["value"],
                "severity":     severity,
                "actions":      contextualised,
            })
        return plan

    # ── PUBLIC: generate_plan ─────────────────
    def generate_plan(self, student_id: str) -> dict:
        """
        Main entry point.  Returns:
        {
          student_id: str,
          cluster_label: str,
          placement_prob: float,
          risk_label: int,
          issues: [...],
          action_plan: [...],
          summary: str
        }
        """
        record         = self._observe(student_id)
        issues         = self._reason(record)
        action_plan    = self._act(issues)
        placement_prob = float(record.get("placement_prob", 0.5))
        cluster_label  = str(record.get("cluster_label", "Unknown"))
        risk_label     = int(record.get("Risk_Label", 0))

        # Dynamic summary
        n_critical = sum(1 for i in issues if i["severity"] == "critical")
        if n_critical >= 3:
            summary = (
                f"⚠️ {student_id} has {n_critical} critical gaps. "
                "Immediate, structured intervention is required."
            )
        elif n_critical > 0:
            summary = (
                f"🟡 {student_id} has {n_critical} critical area(s) needing focused effort "
                "alongside moderate improvements in other dimensions."
            )
        elif issues:
            summary = (
                f"🔵 {student_id} is progressing but needs consistent improvement "
                f"in {len(issues)} area(s) to unlock their full placement potential."
            )
        else:
            summary = (
                f"✅ {student_id} is placement-ready. Focus on acing interviews!"
            )

        return {
            "student_id":      student_id,
            "cluster_label":   cluster_label,
            "placement_prob":  round(placement_prob, 4),
            "risk_label":      risk_label,
            "issues":          issues,
            "action_plan":     action_plan,
            "summary":         summary,
        }

    # ── PUBLIC: batch_plans ───────────────────
    def batch_plans(self, student_ids: list[str]) -> list[dict]:
        plans = []
        for sid in student_ids:
            try:
                plans.append(self.generate_plan(sid))
            except ValueError:
                pass
        return plans

    # ── PUBLIC: get_urgent_list ───────────────
    def get_urgent_list(self, top_n: int | None = None) -> pd.DataFrame:
        """
        Returns a DataFrame of the most at-risk students ranked by:
        risk_label → cluster → (1 - placement_prob)
        """
        df = self.raw_df.copy()
        
        # 1. Base filter: Silent/At-Risk cluster OR manually flagged High-Risk
        at_risk = df[(df["cluster_label"] == "Silent/At-Risk") | (df["Risk_Label"] == 1)].copy()
        
        # 2. Strict exclusion: Remove students with high placement prob (>65%)
        # UNLESS they have explicitly been flagged with Risk_Label = 1
        condition = (at_risk["placement_prob"] < 0.65) | (at_risk["Risk_Label"] == 1)
        at_risk = at_risk[condition]
        
        at_risk["urgency_score"] = (
            at_risk["Risk_Label"] * 2
            + (1 - at_risk["placement_prob"])
        )
        at_risk = at_risk.sort_values("urgency_score", ascending=False)
        cols = [
            "student_id", "branch", "cgpa", "coding_skill_score",
            "attendance_percentage", "placement_prob", "Risk_Label",
            "cluster_label", "engagement_score",
        ]
        res = at_risk[[c for c in cols if c in at_risk.columns]]
        return res.head(top_n) if top_n else res


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    metrics = train_and_save()
    art     = load_artefacts()
    df      = art["df"]

    agent   = InterventionAgent(raw_df=df)
    urgent  = agent.get_urgent_list(top_n=5)

    if not urgent.empty:
        sid  = urgent.iloc[0]["student_id"]
        plan = agent.generate_plan(sid)
        print(f"\n── Action Plan for {sid} ──────────────────")
        print(f"Cluster        : {plan['cluster_label']}")
        print(f"Placement Prob : {plan['placement_prob']:.2%}")
        print(f"Summary        : {plan['summary']}")
        for item in plan["action_plan"]:
            print(f"\n  [{item['severity'].upper()}] {item['dimension']} = {item['current_val']}")
            for a in item["actions"]:
                print(f"    • {a}")
