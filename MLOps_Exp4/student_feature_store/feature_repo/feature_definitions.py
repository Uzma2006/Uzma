
from datetime import timedelta
from feast.value_type import ValueType
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Define student entity
student = Entity(
    name="student",
    join_keys=["student_id"],
    value_type=ValueType.INT64
)

# Define dataset source
student_source = FileSource(
    name="student_source",
    path="data/student_stats.parquet",
    timestamp_field="event_timestamp"
)

# Define student feature view
student_features = FeatureView(
    name="student_features",
    entities=[student],
    ttl=timedelta(days=365),
    schema=[
        Field(name="study_hours", dtype=Float32),
        Field(name="attendance", dtype=Float32),
        Field(name="marks", dtype=Int64),
    ],
    source=student_source,
    online=True
)
