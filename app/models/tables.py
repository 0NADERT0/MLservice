import datetime

from sqlalchemy import MetaData, Table, Column, String, TIMESTAMP, DateTime

metadata = MetaData()

result = Table(
    "result",
    metadata,
    Column("file_name", String),
    Column("file_path", String),
    Column("predict", String),
    Column("predict_file_path", String),
    Column("predict_date", DateTime, default=datetime.datetime.now)
)