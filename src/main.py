import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.bronze_layer import load_bronze
from src.silver_layer import process_silver
from src.gold_layer import process_gold
from src.ml_pipeline import run_ml
from src.delta_maintenance import maintain_silver
from src.config import RAW_PATH

def main():
    print("Starting Lakehouse Pipeline...")
    
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw path {RAW_PATH} not found. Check docker volumes.")
        
    csv_files = list(RAW_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {RAW_PATH}")
        
    print(f"Found {len(csv_files)} file(s).")

    try:
        print("\n[1/5] Bronze Layer...")
        load_bronze()
        
        print("\n[2/5] Silver Layer...")
        process_silver()

        print("\n[3/5] Delta Lake maintenance...")
        maintain_silver()
        
        print("\n[4/5] Gold Layer...")
        process_gold()
        
        print("\n[5/5] ML Pipeline...")
        run_ml()
        
        print("\nDONE!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
