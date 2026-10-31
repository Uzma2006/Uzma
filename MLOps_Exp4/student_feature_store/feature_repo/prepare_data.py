
import pandas as pd

# Create student dataset
data = {
    "student_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance": [60, 65, 70, 80, 90, 95, 98, 100],
    "marks": [35, 40, 50, 65, 75, 85, 90, 95],
    "event_timestamp": pd.to_datetime([
        "2026-10-20",
        "2026-10-21",
        "2026-10-22",
        "2026-10-23",
        "2026-10-24",
        "2026-10-25",
        "2026-10-26",
        "2026-10-27"
    ], utc=True)
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Save as Parquet file
df.to_parquet("data/student_stats.parquet", index=False)

print("Student dataset created successfully!")
print(df)
