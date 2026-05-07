from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "breast-cancer.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "breast-cancer-processed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
TARGET_MAPPING = {"B": NEGATIVE_LABEL, "M": POSITIVE_LABEL}
CLASS_NAMES = {NEGATIVE_LABEL: "Benign", POSITIVE_LABEL: "Malignant"}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
