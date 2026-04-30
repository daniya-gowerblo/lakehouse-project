from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
BRONZE_PATH = DATA_ROOT / "bronze"
SILVER_PATH = DATA_ROOT / "silver"
GOLD_PATH = DATA_ROOT / "gold"
RAW_PATH = DATA_ROOT / "raw"

DELAY_THRESHOLD = 15