import shutil

import polars as pl
from deltalake import write_deltalake

from src.config import GOLD_PATH, SILVER_PATH


def _write_delta(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_deltalake(str(path), df, mode="overwrite")


def _mean(column, alias):
    return pl.col(column).mean().alias(alias)


def _delay_rate():
    return (pl.col("arr_delay") > 15).mean().alias("delay_rate")


def _build_airline_delay(df):
    return (
        df.group_by(["year", "month", "airline"])
        .agg([
            pl.len().alias("flight_count"),
            _mean("arr_delay", "avg_arr_delay"),
            pl.col("arr_delay").median().alias("median_arr_delay"),
            pl.col("arr_delay").quantile(0.95).alias("p95_arr_delay"),
            _delay_rate(),
            _mean("dep_delay", "avg_dep_delay"),
            _mean("distance", "avg_distance"),
        ])
        .sort(["year", "month", "avg_arr_delay"], descending=[False, False, True])
        .collect()
    )


def _build_airport_delay(df):
    origin = (
        df.group_by(["year", "month", "origin"])
        .agg([
            pl.len().alias("flight_count"),
            _mean("dep_delay", "avg_dep_delay"),
            _mean("arr_delay", "avg_arr_delay"),
            _delay_rate(),
        ])
        .rename({"origin": "airport"})
        .with_columns(pl.lit("origin").alias("airport_role"))
    )

    dest = (
        df.group_by(["year", "month", "dest"])
        .agg([
            pl.len().alias("flight_count"),
            _mean("arr_delay", "avg_arr_delay"),
            _delay_rate(),
        ])
        .rename({"dest": "airport"})
        .with_columns([
            pl.lit(None, dtype=pl.Float64).alias("avg_dep_delay"),
            pl.lit("destination").alias("airport_role"),
        ])
        .select(origin.collect_schema().names())
    )

    return pl.concat([origin.collect(), dest.collect()], how="vertical")


def _build_route_delay(df):
    return (
        df.group_by(["year", "month", "origin", "dest", "route"])
        .agg([
            pl.len().alias("flight_count"),
            _mean("arr_delay", "avg_arr_delay"),
            pl.col("arr_delay").median().alias("median_arr_delay"),
            pl.col("arr_delay").quantile(0.95).alias("p95_arr_delay"),
            _delay_rate(),
            _mean("distance", "avg_distance"),
        ])
        .sort(["year", "month", "avg_arr_delay"], descending=[False, False, True])
        .collect()
    )


def _build_time_delay(df):
    return (
        df.group_by(["year", "month", "day_of_week", "hour", "season"])
        .agg([
            pl.len().alias("flight_count"),
            _mean("arr_delay", "avg_arr_delay"),
            pl.col("arr_delay").median().alias("median_arr_delay"),
            _delay_rate(),
            _mean("dep_delay", "avg_dep_delay"),
        ])
        .sort(["year", "month", "day_of_week", "hour"])
        .collect()
    )


def _build_ml_features(df):
    return (
        df.select([
            "flight_date",
            "year",
            "month",
            "airline",
            "origin",
            "dest",
            "route",
            "distance",
            "hour",
            "day_of_week",
            "season",
            "dep_delay",
            "arr_delay",
        ])
        .with_columns([
            (pl.col("arr_delay") > 15).cast(pl.Int8).alias("is_delayed"),
            pl.len().over(["flight_date", "origin"]).cast(pl.Int32).alias("origin_day_flights"),
            pl.len().over(["flight_date", "dest"]).cast(pl.Int32).alias("dest_day_flights"),
            pl.len().over(["flight_date", "airline"]).cast(pl.Int32).alias("airline_day_flights"),
            pl.len().over(["flight_date", "route"]).cast(pl.Int32).alias("route_day_flights"),
            pl.len().over(["flight_date", "origin", "hour"]).cast(pl.Int32).alias("origin_hour_flights"),
            pl.len().over(["flight_date", "route", "hour"]).cast(pl.Int32).alias("route_hour_flights"),
        ])
        .collect()
    )


def process_gold():
    df = pl.scan_delta(str(SILVER_PATH))

    analytics_path = GOLD_PATH / "analytics"
    if (analytics_path / "_delta_log").exists():
        shutil.rmtree(analytics_path)

    _write_delta(analytics_path / "airline_delay", _build_airline_delay(df))
    _write_delta(analytics_path / "airport_delay", _build_airport_delay(df))
    _write_delta(analytics_path / "route_delay", _build_route_delay(df))
    _write_delta(analytics_path / "time_delay", _build_time_delay(df))

    ml_data = _build_ml_features(df)
    _write_delta(GOLD_PATH / "ml_features", ml_data)
    print("  Gold analytics marts and ML feature table created.")
