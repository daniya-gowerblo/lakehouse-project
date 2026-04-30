import polars as pl
from deltalake import DeltaTable, write_deltalake
from src.config import SILVER_PATH, BRONZE_PATH

def process_silver():
    print("  Reading Bronze...")
    df = pl.scan_delta(str(BRONZE_PATH))
    
    df_clean = (
        df
        .filter(
            (pl.col("Cancelled") == 0) & 
            (pl.col("ArrDelay").is_not_null())
        )
        .with_columns([
            (pl.col("CRSDepTime") // 100).cast(pl.Int8).alias("hour"),
            pl.when(pl.col("Month").is_in([12, 1, 2])).then(pl.lit("Winter"))
             .when(pl.col("Month").is_in([3, 4, 5])).then(pl.lit("Spring"))
             .when(pl.col("Month").is_in([6, 7, 8])).then(pl.lit("Summer"))
             .otherwise(pl.lit("Autumn")).alias("season"),
            pl.concat_str([pl.col("Origin"), pl.lit("-"), pl.col("Dest")]).alias("route")
        ])
        .rename({"Year": "year", "Month": "month"})
        .select([
            "FlightDate", "year", "month", "DayOfWeek",
            "IATA_Code_Marketing_Airline", "Flight_Number_Marketing_Airline",
            "Origin", "Dest", "CRSDepTime", "DepDelay",
            "CRSArrTime", "ArrDelay", "Distance",
            "hour", "season", "route"
        ])
    )
    
    df_new = df_clean.collect()
    
    if df_new.is_empty():
        print("  No data after filtering.")
        return

    merge_keys = ["FlightDate", "IATA_Code_Marketing_Airline", "Flight_Number_Marketing_Airline", "Origin", "Dest"]
    df_new = df_new.unique(subset=merge_keys, keep="first")
    print(f"  Deduplicated source data. Rows: {len(df_new)}")

    SILVER_PATH.mkdir(parents=True, exist_ok=True)
    
    if DeltaTable.is_deltatable(str(SILVER_PATH)):
        dt = DeltaTable(str(SILVER_PATH))
        
        predicate = " AND ".join([f"target.{k} = source.{k}" for k in merge_keys])
        
        dt.merge(
            source=df_new, 
            predicate=predicate, 
            source_alias="source", 
            target_alias="target"
        ).when_matched_update_all().when_not_matched_insert_all().execute()
        print("  Silver merged.")
    else:
        write_deltalake(
            str(SILVER_PATH), 
            df_new, 
            partition_by=["year", "month"], 
            mode="overwrite"
        )
        print("  Silver created.")
        
    try:
        DeltaTable(str(SILVER_PATH)).vacuum(retention_hours=168) # 7 дней, чтобы избежать ошибки вакуума
    except Exception as e:
        pass 