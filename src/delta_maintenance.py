from pathlib import Path

import polars as pl
from deltalake import DeltaTable

from src.config import LOGS_PATH, SILVER_PATH


def maintain_silver():
    if not DeltaTable.is_deltatable(str(SILVER_PATH)):
        print("  Silver maintenance skipped: table does not exist.")
        return

    dt = DeltaTable(str(SILVER_PATH))

    print("  Delta OPTIMIZE compact...")
    compact_metrics = dt.optimize.compact()
    print(f"  Compact metrics: {compact_metrics}")

    dt = DeltaTable(str(SILVER_PATH))
    print("  Delta Z-ORDER by origin, dest, hour...")
    z_order_metrics = dt.optimize.z_order(["origin", "dest", "hour"])
    print(f"  Z-ORDER metrics: {z_order_metrics}")

    dt = DeltaTable(str(SILVER_PATH))
    deleted_files = dt.vacuum(retention_hours=168, dry_run=False)
    print(f"  VACUUM deleted files: {len(deleted_files)}")

    _write_time_travel_sample()


def _write_time_travel_sample():
    current = DeltaTable(str(SILVER_PATH))
    current_version = current.version()
    if current_version == 0:
        print("  Time travel skipped: only version 0 exists.")
        return

    previous_version = current_version - 1
    try:
        sample = pl.read_delta(str(SILVER_PATH), version=previous_version).head(5)
    except Exception as exc:
        print(f"  Time travel sample skipped for Silver version {previous_version}: {exc}")
        return

    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = Path(LOGS_PATH) / "silver_time_travel_previous_version.csv"
    sample.write_csv(output_path)
    print(f"  Time travel sample saved for Silver version {previous_version}: {output_path}")
