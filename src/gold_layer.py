import polars as pl
from deltalake import write_deltalake
from src.config import GOLD_PATH, SILVER_PATH

def process_gold():
    df = pl.scan_delta(str(SILVER_PATH))
    
    agg = (
        df.group_by(["IATA_Code_Marketing_Airline", "Origin", "season"])
        .agg([
            pl.col("ArrDelay").mean().alias("avg_arr_delay"),
            pl.count().alias("count")
        ])
        .collect()
    )
    
    GOLD_PATH.mkdir(parents=True, exist_ok=True)
    write_deltalake(str(GOLD_PATH / "analytics"), agg, mode="overwrite")
    
    ml_data = (
        df.select([
            "IATA_Code_Marketing_Airline", "Origin", "Dest", "Distance",
            "hour", "season", "DepDelay", "ArrDelay"
        ])
        .with_columns([(pl.col("ArrDelay") > 15).cast(pl.Int8).alias("is_delayed")])
        .collect()
    )

    write_deltalake(str(GOLD_PATH / "ml_features"), ml_data, mode="overwrite")
    print("  Gold layers created.")