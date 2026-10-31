import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Create entity DataFrame
entity_df = pd.DataFrame({
    "student_id": [1, 2],
    "event_timestamp": pd.to_datetime(
        ["2026-10-27T00:00:00Z", "2026-10-27T00:00:00Z"],
        utc=True
    )
})

# Retrieve historical features
historical_features = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "student_features:study_hours",
        "student_features:attendance",
        "student_features:marks"
    ]
).to_df()

print("Historical Features:")
print(historical_features)
# Retrieve online features
online_features = store.get_online_features(
    features=[
        "student_features:study_hours",
        "student_features:attendance",
        "student_features:marks"
    ],
    entity_rows=[
        {"student_id": 1},
        {"student_id": 2}
    ]
).to_dict()

print("\nOnline Features:")
print(online_features)