"""
data_processor.py
-----------------
Silent & At-Risk Student Detection System
Data Pipeline: Loads, engineers features, and prepares the
Student Placement Prediction Dataset 2026.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ENGAGEMENT_WEIGHTS = {
    "attendance_percentage": 0.45,
    "coding_skill_score": 0.55,
}
HIGH_CGPA_THRESHOLD = 7.5          # CGPA considered "high"
RISK_LABEL_COL      = "Risk_Label" # 1 = at-risk, 0 = not at-risk

# Default dataset path (auto-detect in project directory)
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_placement_prediction_dataset_2026.csv",
)

CATEGORICAL_COLS = [
    "branch", "gender", "internship_experience",
    "backlogs", "placement_status",
]

FEATURE_COLS = [
    "cgpa",
    "attendance_percentage",
    "coding_skill_score",
    "communication_skill_score",
    "technical_skill_score",
    "projects_count",
    "certifications_count",
    "internship_experience",
    "backlogs",
    "engagement_score",
    "gender",
    "branch",
]


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (fallback if CSV absent)
# ─────────────────────────────────────────────
def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset that mirrors the
    Student Placement Prediction Dataset 2026 schema.
    Used when the real CSV is not available locally.
    """
    rng = np.random.default_rng(seed)
    branches = ["CSE", "ECE", "ME", "CE", "IT", "EEE"]
    genders   = ["Male", "Female"]

    cgpa        = np.clip(rng.normal(7.2, 0.9, n), 4.0, 10.0)
    attend      = np.clip(rng.normal(78, 12, n), 30, 100)
    code_skill  = np.clip(rng.normal(65, 18, n), 10, 100)
    comm_skill  = np.clip(rng.normal(68, 15, n), 10, 100)
    tech_skill  = np.clip(rng.normal(66, 17, n), 10, 100)
    projects    = rng.integers(0, 8, n)
    certs       = rng.integers(0, 6, n)
    internship  = rng.choice([0, 1], n, p=[0.45, 0.55])
    backlogs    = rng.choice([0, 1], n, p=[0.75, 0.25])
    branch      = rng.choice(branches, n)
    gender      = rng.choice(genders, n)

    # Placement logic: higher chance with high cgpa, coding, internship
    score = (
        0.25 * (cgpa / 10)
        + 0.25 * (code_skill / 100)
        + 0.20 * (tech_skill / 100)
        + 0.15 * internship
        + 0.10 * (certs / 6)
        + 0.05 * (1 - backlogs)
    )
    placed_prob   = np.clip(score + rng.normal(0, 0.08, n), 0, 1)
    placement_status = np.where(placed_prob >= 0.52, "Placed", "Not Placed")

    df = pd.DataFrame({
        "student_id":                [f"STU{1000+i}" for i in range(n)],
        "branch":                    branch,
        "gender":                    gender,
        "cgpa":                      np.round(cgpa, 2),
        "attendance_percentage":     np.round(attend, 1),
        "coding_skill_score":        np.round(code_skill, 1),
        "communication_skill_score": np.round(comm_skill, 1),
        "technical_skill_score":     np.round(tech_skill, 1),
        "projects_count":            projects,
        "certifications_count":      certs,
        "internship_experience":     internship,
        "backlogs":                  backlogs,
        "placement_status":          placement_status,
    })
    return df


# ─────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────
class StudentDataProcessor:
    """
    End-to-end data processor for the Silent & At-Risk Detection System.

    Steps
    -----
    1. load()          – read CSV or fall back to synthetic data
    2. clean()         – handle missing values and types
    3. engineer()      – create engagement_score and Risk_Label
    4. encode()        – label-encode categoricals
    5. scale()         – Min-Max scale numeric features
    6. get_features()  – return X, y ready for modelling
    """

    def __init__(self, csv_path: str | None = None):
        # Default to the real dataset bundled with the project
        if csv_path is None:
            csv_path = DEFAULT_CSV
        self.csv_path  = csv_path
        self.df_raw    = None
        self.df        = None
        self.le_dict   = {}
        self.scaler    = MinMaxScaler()
        self._fitted   = False

    # ── 1. LOAD ──────────────────────────────
    def load(self) -> "StudentDataProcessor":
        if self.csv_path and os.path.exists(self.csv_path):
            self.df_raw = pd.read_csv(self.csv_path)
            print(f"[DataProcessor] Loaded {len(self.df_raw)} rows from {self.csv_path}")
        else:
            print("[DataProcessor] CSV not found – using synthetic data.")
            self.df_raw = generate_synthetic_data()

        # Standardise column names in raw df too
        self.df_raw.columns = [c.strip().lower().replace(" ", "_") for c in self.df_raw.columns]

        # Ensure student_id is string in raw df
        if "student_id" in self.df_raw.columns:
            self.df_raw["student_id"] = self.df_raw["student_id"].astype(str)

        self.df = self.df_raw.copy()
        return self

    # ── 2. CLEAN ─────────────────────────────
    def clean(self) -> "StudentDataProcessor":
        df = self.df
        # Standardise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # ── Column mapping for the real dataset ─────────
        # The real CSV uses 'internships_count' instead of 'internship_experience'
        if "internships_count" in df.columns and "internship_experience" not in df.columns:
            df["internship_experience"] = (df["internships_count"] > 0).astype(int)
            print("[DataProcessor] Mapped internships_count -> internship_experience (binary)")

        # The real CSV lacks 'technical_skill_score' but has aptitude + logical reasoning
        if "technical_skill_score" not in df.columns:
            if "aptitude_score" in df.columns and "logical_reasoning_score" in df.columns:
                df["technical_skill_score"] = (
                    0.6 * df["aptitude_score"] + 0.4 * df["logical_reasoning_score"]
                )
                print("[DataProcessor] Derived technical_skill_score = 0.6*aptitude + 0.4*logical_reasoning")
            elif "aptitude_score" in df.columns:
                df["technical_skill_score"] = df["aptitude_score"]
                print("[DataProcessor] Mapped aptitude_score -> technical_skill_score")
            elif "logical_reasoning_score" in df.columns:
                df["technical_skill_score"] = df["logical_reasoning_score"]
                print("[DataProcessor] Mapped logical_reasoning_score -> technical_skill_score")
            else:
                df["technical_skill_score"] = 50.0  # neutral fallback
                print("[DataProcessor] WARNING: No source for technical_skill_score, set to 50.0")

        # Convert backlogs to binary (0 = no backlogs, 1 = has backlogs)
        if "backlogs" in df.columns and df["backlogs"].dtype in [np.int64, np.float64, int, float]:
            if df["backlogs"].max() > 1:
                df["backlogs"] = (df["backlogs"] > 0).astype(int)
                print("[DataProcessor] Binarised backlogs column")

        # Convert student_id to string if numeric
        if "student_id" in df.columns:
            df["student_id"] = df["student_id"].astype(str)
        else:
            df["student_id"] = [f"STU{i}" for i in range(len(df))]

        # Ensure required columns exist
        required = [
            "cgpa", "attendance_percentage", "coding_skill_score",
            "communication_skill_score", "technical_skill_score",
            "placement_status",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[DataProcessor] Missing columns: {missing}")

        # Fill numeric NaNs with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())

        # Fill categorical NaNs with mode
        cat_cols = df.select_dtypes(include="object").columns
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0])

        # Clip skill scores to [0, 100]
        skill_cols = [c for c in df.columns if "score" in c or "percentage" in c]
        for c in skill_cols:
            df[c] = df[c].clip(0, 100)

        self.df = df
        return self

    # ── 3. FEATURE ENGINEERING ───────────────
    def engineer(self) -> "StudentDataProcessor":
        df = self.df

        # Engagement Score (weighted average)
        df["engagement_score"] = (
            ENGAGEMENT_WEIGHTS["attendance_percentage"] * df["attendance_percentage"]
            + ENGAGEMENT_WEIGHTS["coding_skill_score"]  * df["coding_skill_score"]
        )

        # Risk Label: Unplaced despite High CGPA
        unplaced     = df["placement_status"].str.lower().str.contains("not placed|unplaced")
        high_cgpa    = df["cgpa"] >= HIGH_CGPA_THRESHOLD
        df[RISK_LABEL_COL] = (unplaced & high_cgpa).astype(int)

        # Placement binary target
        df["placed"] = (~unplaced).astype(int)

        self.df = df
        return self

    # ── 4. ENCODE ────────────────────────────
    def encode(self) -> "StudentDataProcessor":
        df = self.df
        encode_cols = [c for c in CATEGORICAL_COLS if c in df.columns and c != "placement_status"]

        for c in encode_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            self.le_dict[c] = le

        self.df = df
        return self

    # ── 5. SCALE ─────────────────────────────
    def scale(self) -> "StudentDataProcessor":
        self.df_unscaled = self.df.copy()
        feat_present = [c for c in FEATURE_COLS if c in self.df.columns]
        self.df[feat_present] = self.scaler.fit_transform(self.df[feat_present])
        self._fitted = True
        return self

    # ── 6. GET FEATURES ──────────────────────
    def get_features(self) -> tuple[pd.DataFrame, pd.Series]:
        if not self._fitted:
            raise RuntimeError("Call .scale() before .get_features()")
        feat_present = [c for c in FEATURE_COLS if c in self.df.columns]
        X = self.df[feat_present]
        y = self.df["placed"]
        return X, y

    # ── FULL PIPELINE ─────────────────────────
    def run(self) -> "StudentDataProcessor":
        return self.load().clean().engineer().encode().scale()

    # ── HELPERS ──────────────────────────────
    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def get_unscaled_dataframe(self) -> pd.DataFrame:
        if hasattr(self, 'df_unscaled'):
            return self.df_unscaled.copy()
        return self.df.copy()

    def get_raw(self) -> pd.DataFrame:
        return self.df_raw.copy()

    def get_branch_map(self) -> dict:
        """Return int→branch label mapping (for display)."""
        if "branch" in self.le_dict:
            le = self.le_dict["branch"]
            return dict(enumerate(le.classes_))
        return {}


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    proc = StudentDataProcessor()
    proc.run()
    X, y = proc.get_features()
    df   = proc.get_dataframe()

    print("\n── Feature Matrix ──────────────────────")
    print(X.head())
    print(f"\nShape  : {X.shape}")
    print(f"Target : {y.value_counts().to_dict()}")
    print(f"\nRisk Labels: {df['Risk_Label'].value_counts().to_dict()}")
    print(f"Engagement Score (sample):\n{df['engagement_score'].describe()}")
