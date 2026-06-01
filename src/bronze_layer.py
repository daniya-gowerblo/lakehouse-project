import polars as pl
from deltalake import write_deltalake
from datetime import datetime
from src.config import BRONZE_PATH, RAW_PATH

def load_bronze():
    BRONZE_PATH.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(RAW_PATH.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_PATH}")

    print(f"Processing {len(csv_files)} file(s)...")

    for file_path in csv_files:
        print(f"  -> Reading {file_path.name}")
        
        try:
            df = pl.read_csv(file_path, try_parse_dates=True, ignore_errors=True)
            
            if df.is_empty():
                print(f"  -> Skipping empty file: {file_path.name}")
                continue
                
            df = df.with_columns([
                pl.lit(file_path.name).alias("source_file"),
                pl.lit(datetime.now()).alias("ingestion_timestamp")
            ])
            
            table_exists = (BRONZE_PATH / "_delta_log").exists()
            mode = "append" if table_exists else "overwrite"
            
            print(f"  -> Writing to Delta (mode={mode})...")
            
            write_deltalake(
                table_or_uri=str(BRONZE_PATH),
                data=df,
                mode=mode,
                schema_mode="merge"
            )
            
            print(f"  -> Loaded {file_path.name} into Bronze")
            
        except Exception as e:
            print(f"  -> Error processing {file_path.name}: {e}")
            raise e
            
    print("Bronze layer loading complete.")
