"""VisionX Database Layer supporting both SQLite (Local Edge/Telemedicine) and In-Memory storage.

Default: SQLite at data/runtime/telemedicine.db for zero-dependency, local, offline-first deployment.
Clean seams provide an in-memory storage implementation for isolated testing or diskless environments.
"""

from __future__ import annotations

import contextlib
import copy
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Generator
import uuid

import bcrypt

from .config import settings

logger = logging.getLogger("visionx.database")


def hash_pw(pwd: str) -> str:
    pwd_bytes = pwd.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def _json_serializer(val: Any) -> str:
    """Custom JSON serializer ensuring datetime objects format as ISO strings."""
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    return str(val)


def serialize_for_storage(obj: Any) -> Any:
    """Recursively converts datetimes to ISO-8601 strings, UUIDs to str, etc."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {k: serialize_for_storage(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_storage(v) for v in obj]
    return obj


def doc_matches_query(doc: dict, query: dict | None) -> bool:
    """Predicate matching a document against a simple MongoDB-style query."""
    if not query:
        return True
    for k, v in query.items():
        if k == "_id":
            if str(doc.get("_id")) != str(v):
                return False
        elif doc.get(k) != v:
            return False
    return True


class InsertResult:
    """Represents the outcome of an insert operation."""

    def __init__(self, inserted_id: str):
        self.inserted_id = str(inserted_id)


class AsyncCursor:
    """Async-iterable cursor compatible with Motor/MongoDB async cursors."""

    def __init__(self, items: list[dict]):
        self._items = list(items)
        self._index = 0

    def sort(self, key_or_list: str | list[tuple[str, int]], direction: int = 1) -> AsyncCursor:
        if isinstance(key_or_list, list):
            for k, d in reversed(key_or_list):
                self._items.sort(
                    key=lambda x: (x.get(k) is not None, str(x.get(k) or "")),
                    reverse=(d == -1),
                )
        elif isinstance(key_or_list, str):
            self._items.sort(
                key=lambda x: (x.get(key_or_list) is not None, str(x.get(key_or_list) or "")),
                reverse=(direction == -1),
            )
        return self

    def limit(self, n: int) -> AsyncCursor:
        self._items = self._items[:n]
        return self

    def __aiter__(self) -> AsyncCursor:
        self._index = 0
        return self

    async def __anext__(self) -> dict:
        if self._index < len(self._items):
            item = copy.deepcopy(self._items[self._index])
            self._index += 1
            return item
        raise StopAsyncIteration

    async def to_list(self, length: int | None = None) -> list[dict]:
        if length is not None:
            return [copy.deepcopy(x) for x in self._items[:length]]
        return [copy.deepcopy(x) for x in self._items]


# Backward-compatibility alias
AsyncSQLiteCursor = AsyncCursor


class InMemoryCollection:
    """Async in-memory collection compatible with MongoDB collection interface."""

    def __init__(self, name: str):
        self.name = name
        self._docs: dict[str, dict] = {}

    async def find_one(self, query: dict | None = None, sort: list[tuple[str, int]] | None = None) -> dict | None:
        matched = [d for d in self._docs.values() if doc_matches_query(d, query)]
        if not matched:
            return None
        if sort:
            for k, d in reversed(sort):
                matched.sort(
                    key=lambda x: (x.get(k) is not None, str(x.get(k) or "")),
                    reverse=(d == -1),
                )
        return copy.deepcopy(matched[0])

    def find(self, query: dict | None = None) -> AsyncCursor:
        matched = [d for d in self._docs.values() if doc_matches_query(d, query)]
        return AsyncCursor(matched)

    async def insert_one(self, document: dict) -> InsertResult:
        doc = serialize_for_storage(copy.deepcopy(document))
        if "_id" not in doc:
            doc["_id"] = str(uuid.uuid4())
        else:
            doc["_id"] = str(doc["_id"])
        doc_id = str(doc["_id"])
        self._docs[doc_id] = doc
        return InsertResult(doc["_id"])

    async def insert_many(self, documents: list[dict]) -> list[str]:
        ids = []
        for doc in documents:
            res = await self.insert_one(doc)
            ids.append(res.inserted_id)
        return ids

    async def update_one(self, filter: dict, update: dict, upsert: bool = False) -> None:
        matched = [d for d in self._docs.values() if doc_matches_query(d, filter)]
        if matched:
            target = matched[0]
            if "$set" in update:
                target.update(serialize_for_storage(copy.deepcopy(update["$set"])))
            else:
                target.update(serialize_for_storage(copy.deepcopy(update)))
        elif upsert:
            new_doc = copy.deepcopy(filter)
            if "$set" in update:
                new_doc.update(update["$set"])
            else:
                new_doc.update(update)
            await self.insert_one(new_doc)

    async def count_documents(self, filter: dict | None = None) -> int:
        return sum(1 for d in self._docs.values() if doc_matches_query(d, filter))

    async def create_index(self, key_or_keys: Any, unique: bool = False) -> None:
        pass


class InMemoryDatabase:
    """Pure in-memory database with async MongoDB-compatible interface."""

    def __init__(self):
        self._collections: dict[str, InMemoryCollection] = {}

    def __getitem__(self, name: str) -> InMemoryCollection:
        if name not in self._collections:
            self._collections[name] = InMemoryCollection(name)
        return self._collections[name]

    def close(self) -> None:
        self._collections.clear()


class AsyncSQLiteCollection:
    """Async collection interface backed by SQLite."""

    def __init__(self, db: SQLiteDatabase, name: str):
        self.db = db
        self.name = name

    def _get_all_docs(self) -> list[dict]:
        with self.db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM kv_documents WHERE collection = ?",
                (self.name,),
            )
            rows = cursor.fetchall()
            docs = []
            for (data_str,) in rows:
                try:
                    docs.append(json.loads(data_str))
                except Exception:
                    pass
            return docs

    async def find_one(self, query: dict | None = None, sort: list[tuple[str, int]] | None = None) -> dict | None:
        docs = self._get_all_docs()
        matched = [d for d in docs if doc_matches_query(d, query)]
        if not matched:
            return None
        if sort:
            for k, d in reversed(sort):
                matched.sort(
                    key=lambda x: (x.get(k) is not None, str(x.get(k) or "")),
                    reverse=(d == -1),
                )
        return copy.deepcopy(matched[0])

    def find(self, query: dict | None = None) -> AsyncCursor:
        docs = self._get_all_docs()
        matched = [d for d in docs if doc_matches_query(d, query)]
        return AsyncCursor(matched)

    async def insert_one(self, document: dict) -> InsertResult:
        doc = serialize_for_storage(copy.deepcopy(document))
        if "_id" not in doc:
            doc["_id"] = str(uuid.uuid4())
        else:
            doc["_id"] = str(doc["_id"])

        doc_id = str(doc["_id"])
        # If document has email or patient_id, use as unique key when possible
        doc_key = doc.get("email") or doc.get("patient_id") or doc_id
        now_iso = datetime.now(timezone.utc).isoformat()
        if "created_at" not in doc:
            doc["created_at"] = now_iso

        with self.db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO kv_documents (collection, doc_id, doc_key, data, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.name, doc_id, str(doc_key), json.dumps(doc, default=_json_serializer), now_iso),
            )

        return InsertResult(doc["_id"])

    async def insert_many(self, documents: list[dict]) -> list[str]:
        inserted_ids = []
        for doc in documents:
            res = await self.insert_one(doc)
            inserted_ids.append(res.inserted_id)
        return inserted_ids

    async def update_one(self, filter: dict, update: dict, upsert: bool = False) -> None:
        docs = self._get_all_docs()
        matched = [d for d in docs if doc_matches_query(d, filter)]

        if matched:
            target = matched[0]
            if "$set" in update:
                target.update(serialize_for_storage(copy.deepcopy(update["$set"])))
            else:
                target.update(serialize_for_storage(copy.deepcopy(update)))

            doc_id = str(target.get("_id", uuid.uuid4()))
            doc_key = target.get("email") or target.get("patient_id") or doc_id

            with self.db.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO kv_documents (collection, doc_id, doc_key, data, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self.name,
                        doc_id,
                        str(doc_key),
                        json.dumps(target, default=_json_serializer),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        elif upsert:
            new_doc = copy.deepcopy(filter)
            if "$set" in update:
                new_doc.update(update["$set"])
            else:
                new_doc.update(update)
            await self.insert_one(new_doc)

    async def count_documents(self, filter: dict | None = None) -> int:
        docs = self._get_all_docs()
        matched = [d for d in docs if doc_matches_query(d, filter)]
        return len(matched)

    async def create_index(self, key_or_keys: Any, unique: bool = False) -> None:
        pass


class SQLiteDatabase:
    """Lightweight SQLite database with async MongoDB-compatible collection interface."""

    def __init__(self, db_path: Path | str):
        self.is_memory = str(db_path) == ":memory:"
        self.db_path = ":memory:" if self.is_memory else Path(db_path)
        if not self.is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._mem_conn: sqlite3.Connection | None = None
            self._lock: threading.Lock | None = None
        else:
            self._lock = threading.Lock()
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        self._collections: dict[str, AsyncSQLiteCollection] = {}

    @contextlib.contextmanager
    def get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides a managed SQLite connection that cleanly closes/commits on exit."""
        if self.is_memory:
            assert self._mem_conn is not None and self._lock is not None
            with self._lock:
                yield self._mem_conn
                self._mem_conn.commit()
        else:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def __getitem__(self, name: str) -> AsyncSQLiteCollection:
        if name not in self._collections:
            self._collections[name] = AsyncSQLiteCollection(self, name)
        return self._collections[name]

    def close(self) -> None:
        """Closes persistent database connections."""
        if self._mem_conn is not None:
            try:
                self._mem_conn.close()
            except Exception:
                pass
            self._mem_conn = None
        self._collections.clear()


def init_sqlite_schema(conn: sqlite3.Connection) -> None:
    """Creates initial database tables and indexes."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS kv_documents (
            collection TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            doc_key TEXT,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (collection, doc_id)
        );
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_collection_key ON kv_documents (collection, doc_key);"
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            phc_center TEXT,
            created_at TEXT
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS doctor_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            ai_grade INTEGER,
            ai_confidence REAL,
            ai_referable_risk REAL,
            csme_threat INTEGER,
            min_fovea_dist_dd REAL,
            macular_threat TEXT,
            triage_level TEXT,
            doctor_grade INTEGER,
            is_overridden INTEGER DEFAULT 0,
            doctor_notes TEXT,
            dispatch_status TEXT,
            reviewed_at TEXT
        );
        """
    )


db: SQLiteDatabase | InMemoryDatabase | None = None
client: Any = None


def get_db() -> SQLiteDatabase | InMemoryDatabase | None:
    return db


async def connect_to_db() -> None:
    """Initializes the database connection (SQLite by default, with InMemory fallback)."""
    global db, client

    try:
        sqlite_path = settings.DATABASE_PATH
        sqlite_db = SQLiteDatabase(sqlite_path)

        with sqlite_db.get_conn() as conn:
            init_sqlite_schema(conn)

        db = sqlite_db
        logger.info(f"Connected to persistent SQLite database at {sqlite_path}")
        await seed_default_data()

    except Exception as e:
        logger.error(f"Failed to initialize SQLite database ({e}), falling back to in-memory database.")
        db = InMemoryDatabase()
        await seed_default_data()


async def close_db_connection() -> None:
    """Closes database connections cleanly."""
    global db, client
    if db is not None:
        if hasattr(db, "close"):
            db.close()
        db = None
    logger.info("Closed database connection.")


# Aliases for backwards compatibility with main.py and existing scripts
connect_to_mongo = connect_to_db
close_mongo_connection = close_db_connection


async def seed_default_data():
    """Initializes default seed accounts and triage queue if empty."""
    if db is None:
        return

    users_col = db["users"]
    patients_col = db["patients"]
    reviews_col = db["reviews"]

    # 1. Seed demo clinical personas
    demo_users = [
        {
            "email": "doctor@visionx.health",
            "name": "Dr. Ramesh Gupta, MS (Ophthalmology)",
            "hashed_password": hash_pw("Password123!"),
            "role": "specialist",
            "center": "District Hospital Eye Center, Nanded",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "email": "worker@visionx.health",
            "name": "Priya Sharma, Lead Health Worker (ANM)",
            "hashed_password": hash_pw("Password123!"),
            "role": "phc_worker",
            "center": "Rural Primary Health Centre (PHC-04), Baramati",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "email": "patient@visionx.health",
            "name": "Anand Verma (Patient ID: RUR-MH-2049)",
            "hashed_password": hash_pw("Password123!"),
            "role": "patient",
            "center": "Rural PHC Baramati",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]

    for user in demo_users:
        existing = await users_col.find_one({"email": user["email"]})
        if not existing:
            await users_col.insert_one(user)
            logger.info(f"Seeded demo user: {user['email']} ({user['role']})")
        else:
            await users_col.update_one(
                {"email": user["email"]},
                {"$set": {"hashed_password": user["hashed_password"], "name": user["name"], "role": user["role"]}}
            )

    # 2. Seed default rural patients in triage queue if empty
    patient_count = await patients_col.count_documents({})
    if patient_count == 0:
        seed_patients = [
            {
                "patient_id": "RUR-MH-2049",
                "name": "Anand Verma",
                "full_name": "Anand Verma",
                "age": 58,
                "gender": "Male",
                "phc_center": "Rural Primary Health Centre (PHC-04), Baramati",
                "diabetes_duration": "12",
                "visual_acuity": "6/18",
                "systolic_bp": 148,
                "diastolic_bp": 92,
                "hba1c": 8.9,
                "intake_date": datetime.now(timezone.utc).isoformat(),
                "triage_status": "URGENT_REFERRAL",
                "urgency": "HIGH",
                "status": "Pending Specialist Review",
                "latest_grade": 2,
                "latest_csme": True,
                "cqi": 0.88,
                "quality_status": "GRADEABLE_ACCEPT",
                "analysis_summary": {
                    "grade": 2,
                    "is_referable": True,
                    "csme_threat": True,
                    "quality": "GRADEABLE_ACCEPT",
                },
            },
            {
                "patient_id": "RUR-MH-1092",
                "name": "Sunita Deshmukh",
                "full_name": "Sunita Deshmukh",
                "age": 54,
                "gender": "Female",
                "phc_center": "Rural Primary Health Centre (PHC-04), Baramati",
                "diabetes_duration": "4",
                "visual_acuity": "6/6",
                "systolic_bp": 122,
                "diastolic_bp": 78,
                "hba1c": 6.8,
                "intake_date": datetime.now(timezone.utc).isoformat(),
                "triage_status": "ROUTINE_MONITORING",
                "urgency": "ROUTINE",
                "status": "Screened Normal",
                "latest_grade": 0,
                "latest_csme": False,
                "cqi": 0.94,
                "quality_status": "GRADEABLE_ACCEPT",
                "analysis_summary": {
                    "grade": 0,
                    "is_referable": False,
                    "csme_threat": False,
                    "quality": "GRADEABLE_ACCEPT",
                },
            },
            {
                "patient_id": "RUR-MH-4421",
                "name": "Baburao Shinde",
                "full_name": "Baburao Shinde",
                "age": 63,
                "gender": "Male",
                "phc_center": "Sub-District Health Centre, Karad",
                "diabetes_duration": "16",
                "visual_acuity": "6/24",
                "systolic_bp": 160,
                "diastolic_bp": 96,
                "hba1c": 9.8,
                "intake_date": datetime.now(timezone.utc).isoformat(),
                "triage_status": "URGENT_REFERRAL",
                "urgency": "HIGH",
                "status": "Specialist Review Required",
                "latest_grade": 3,
                "latest_csme": True,
                "cqi": 0.82,
                "quality_status": "GRADEABLE_ACCEPT",
                "analysis_summary": {
                    "grade": 3,
                    "is_referable": True,
                    "csme_threat": True,
                    "quality": "GRADEABLE_ACCEPT",
                },
            },
        ]
        await patients_col.insert_many(seed_patients)
        logger.info("Seeded initial rural PHC triage queue patients.")

    # 3. Seed initial review in reviews collection if empty
    review_count = await reviews_col.count_documents({})
    if review_count == 0:
        seed_review = {
            "patient_id": "000c1434d8d7",
            "image_id": "000c1434d8d7",
            "ai_grade": 2,
            "ai_suggested_grade": 2,
            "ai_confidence": 88.5,
            "ai_referable_risk": 74.6,
            "csme_threat": True,
            "min_fovea_dist_dd": 0.25,
            "macular_threat": "Macular Threat Confirmed (<1.0 DD)",
            "triage_level": "REFERRAL",
            "doctor_grade": 2,
            "clinician_confirmed_grade": 2,
            "is_overridden": False,
            "agreed_with_ai": True,
            "doctor_notes": "Concur with AI staging. Hard exudate circinate rings within 0.25 DD of fovea center. Urgent OCT and anti-VEGF referral issued.",
            "clinical_notes": "Concur with AI staging. Hard exudate circinate rings within 0.25 DD of fovea center. Urgent OCT and anti-VEGF referral issued.",
            "followup_plan": "Urgent Specialist Referral (14 Days)",
            "dispatch_status": "REFERRAL_DISPATCHED",
            "patient_meta": {
                "patient_name": "Anand Verma",
                "laterality": "OD (Right Eye)",
                "age": 58,
                "gender": "Male",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reviewed_by": "Dr. Ramesh Gupta, MS (Ophthalmology)",
            "reviewer_name": "Dr. Ramesh Gupta, MS (Ophthalmology)",
            "reviewed_by_email": "doctor@visionx.health",
            "clinical_center": "District Hospital Eye Center, Nanded",
        }
        await reviews_col.insert_one(seed_review)
        logger.info("Seeded initial verified doctor review.")
