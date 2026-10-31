from datetime import datetime, timezone
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Load historical data into the online store
store.materialize(
    start_date=datetime(2026, 10, 20, tzinfo=timezone.utc),
    end_date=datetime(2026, 10, 28, tzinfo=timezone.utc)
)

print("Student features loaded into online store successfully!")