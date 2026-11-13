"""Unit tests for VisionX database storage implementations and datetime serialization."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone
import pytest
from services.api.database import (
    InMemoryDatabase,
    SQLiteDatabase,
    init_sqlite_schema,
    serialize_for_storage,
)


@pytest.mark.anyio
async def test_in_memory_database_crud_and_iso_dates():
    db = InMemoryDatabase()
    col = db["test_collection"]

    dt_now = datetime.now(timezone.utc)
    res = await col.insert_one({
        "name": "Test Item",
        "created_at": dt_now,
        "metadata": {"checked_at": dt_now},
    })
    assert res.inserted_id is not None

    found = await col.find_one({"name": "Test Item"})
    assert found is not None
    # Datetimes must be serialized into valid ISO strings
    assert isinstance(found["created_at"], str)
    assert found["created_at"] == dt_now.isoformat()
    assert isinstance(found["metadata"]["checked_at"], str)
    assert found["metadata"]["checked_at"] == dt_now.isoformat()

    # Update
    new_dt = datetime.now(timezone.utc)
    await col.update_one({"name": "Test Item"}, {"$set": {"updated_at": new_dt}})
    updated = await col.find_one({"name": "Test Item"})
    assert updated["updated_at"] == new_dt.isoformat()

    # Cursor sort and limit
    cursor = col.find()
    cursor.sort("created_at", -1).limit(5)
    items = await cursor.to_list()
    assert len(items) == 1

    db.close()


@pytest.mark.anyio
async def test_sqlite_in_memory_mode():
    db = SQLiteDatabase(":memory:")
    with db.get_conn() as conn:
        init_sqlite_schema(conn)

    col = db["patients"]
    dt_now = datetime.now(timezone.utc)
    res = await col.insert_one({
        "patient_id": "P-TEST-001",
        "name": "Patient One",
        "created_at": dt_now,
    })
    assert res.inserted_id is not None

    found = await col.find_one({"patient_id": "P-TEST-001"})
    assert found is not None
    assert found["name"] == "Patient One"
    assert isinstance(found["created_at"], str)
    assert found["created_at"] == dt_now.isoformat()

    count = await col.count_documents({})
    assert count == 1

    db.close()


def test_serialize_for_storage_helper():
    now = datetime(2026, 11, 13, 1, 30, 0, tzinfo=timezone.utc)
    doc = {
        "timestamp": now,
        "nested": {"time": now},
        "list": [now, 123, "hello"],
    }
    serialized = serialize_for_storage(doc)
    assert serialized["timestamp"] == "2026-11-13T01:30:00+00:00"
    assert serialized["nested"]["time"] == "2026-11-13T01:30:00+00:00"
    assert serialized["list"][0] == "2026-11-13T01:30:00+00:00"
