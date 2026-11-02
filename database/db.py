"""
SmartFace Database Layer
========================
Supports both SQLite (local development) and PostgreSQL (Neon production).
Auto-detects based on DATABASE_URL environment variable.
"""

import os
from config import Config


def get_db_connection():
    """Get a database connection. Auto-detects PostgreSQL vs SQLite."""
    if Config.USE_POSTGRES:
        return _get_postgres_connection()
    else:
        return _get_sqlite_connection()


def _get_sqlite_connection():
    """SQLite connection for local development."""
    import sqlite3
    conn = sqlite3.connect(Config.DATABASE_URI)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _get_postgres_connection():
    """PostgreSQL connection for production (Neon).
    Returns a wrapper that behaves identically to sqlite3 connections,
    so ALL existing conn.execute() calls with ? placeholders work as-is.
    """
    import psycopg2
    import psycopg2.extras

    raw_conn = psycopg2.connect(Config.DATABASE_URL, sslmode='require')
    raw_conn.autocommit = False
    return PostgresConnectionWrapper(raw_conn)


class PostgresCursorWrapper:
    """Makes psycopg2 cursor results behave like sqlite3.Row objects."""
    def __init__(self, cursor):
        self._cursor = cursor
        self._description = cursor.description

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return PostgresRowWrapper(self._cursor, row)

    def fetchall(self):
        rows = self._cursor.fetchall()
        return [PostgresRowWrapper(self._cursor, row) for row in rows]

    def __iter__(self):
        return self

    def __next__(self):
        row = self._cursor.fetchone()
        if row is None:
            raise StopIteration
        return PostgresRowWrapper(self._cursor, row)


class PostgresConnectionWrapper:
    """Wraps psycopg2 connection to behave exactly like sqlite3 connection.
    - Auto-converts ? placeholders to %s
    - Returns dict-like Row objects from .execute().fetchone()
    - Supports .executescript() via splitting on ;
    """
    def __init__(self, conn):
        self._conn = conn

    def execute(self, query, params=None):
        # Convert sqlite-style ? to postgres-style %s
        query = query.replace('?', '%s')
        cur = self._conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        return PostgresCursorWrapper(cur)

    def executescript(self, script):
        cur = self._conn.cursor()
        cur.execute(script)
        return cur

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def cursor(self):
        return self._conn.cursor()


class PostgresRowWrapper:
    """Make psycopg2 rows behave like sqlite3.Row (dict-like access)."""
    def __init__(self, cursor, row):
        self._data = {}
        if row and cursor.description:
            for i, col in enumerate(cursor.description):
                self._data[col[0]] = row[i]

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self._data.values())[key]
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


def init_db():
    """Initialize database tables and default data."""
    if Config.USE_POSTGRES:
        _init_postgres()
    else:
        _init_sqlite()


def _init_sqlite():
    """Initialize SQLite database."""
    import sqlite3

    # Use SQLite-specific schema
    sqlite_schema = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        department TEXT,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'employee' CHECK(role IN ('admin', 'employee')),
        profile_photo TEXT,
        face_encoding BLOB,
        face_encodings TEXT,
        face_embedding TEXT,
        face_registered INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('Present', 'Absent', 'Late', 'Leave')),
        method TEXT DEFAULT 'Face Recognition',
        FOREIGN KEY(user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    -- One row per LIVENESS DECISION CHANGE (LIVE/SPOOF/UNCERTAIN), not per
    -- frame -- app.py only inserts when the decision differs from the last
    -- one logged for that session.
    CREATE TABLE IF NOT EXISTS liveness_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        decision TEXT NOT NULL CHECK(decision IN ('LIVE', 'SPOOF', 'UNCERTAIN')),
        p_spoof REAL NOT NULL,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- One row per scan-pipeline STAGE reached per request (deepest stage
    -- only, deduped against the previous frame's stage for the continuous
    -- states -- see app.py's _log_scan_event()). For reviewing a test
    -- session after the fact: what happened, in order, with timestamps.
    CREATE TABLE IF NOT EXISTS scan_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        stage TEXT NOT NULL CHECK(stage IN (
            'NO_FACE', 'MULTI_FACE', 'FACE_MOVED', 'SPOOF', 'UNCERTAIN',
            'RECOGNIZED', 'UNRECOGNIZED', 'CHALLENGE_PASSED',
            'ATTENDANCE_MARKED', 'ALREADY_MARKED'
        )),
        user_id INTEGER,
        detail TEXT,
        test_label TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Performance indexes for 500+ employee scale
    CREATE INDEX IF NOT EXISTS idx_attendance_user_date ON attendance(user_id, date);
    CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
    CREATE INDEX IF NOT EXISTS idx_users_department ON users(department);
    CREATE INDEX IF NOT EXISTS idx_liveness_events_created ON liveness_events(created_at);
    CREATE INDEX IF NOT EXISTS idx_scan_events_created ON scan_events(created_at);
    """

    os.makedirs(os.path.dirname(Config.DATABASE_URI), exist_ok=True)
    conn = _get_sqlite_connection()
    conn.executescript(sqlite_schema)
    _insert_defaults(conn)
    conn.commit()
    conn.close()


def _init_postgres():
    """Initialize PostgreSQL database (Neon)."""
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with open(schema_path, 'r') as f:
        schema = f.read()

    conn = _get_postgres_connection()
    conn.executescript(schema)
    conn.commit()

    _insert_defaults_pg(conn)
    conn.commit()
    conn.close()


_DEFAULT_SETTINGS = {
    'late_cutoff_hour': '9',
    'late_cutoff_minute': '0',
    'company_name': 'Sofzenix Technologies',
    'face_tolerance': '0.45',
    'min_face_samples': '5',
    'liveness_threshold': '0.5',
    'email_enabled': '0',
    'email_trigger_hour': '18',
    'email_trigger_minute': '0',
    'hr_email': '',
}


def _admin_from_env():
    """Reads ADMIN_EMAIL/ADMIN_NAME/ADMIN_PASSWORD from the environment for
    first-run admin creation. No default password is ever hardcoded here --
    if these aren't set, no admin is auto-created (see scripts/create_admin.py
    for interactive creation instead)."""
    email = os.environ.get("ADMIN_EMAIL")
    password = os.environ.get("ADMIN_PASSWORD")
    if not email or not password:
        return None
    return {
        "email": email.strip().lower(),
        "password": password,
        "name": os.environ.get("ADMIN_NAME", "Admin"),
    }


def _insert_defaults(conn):
    """Insert settings defaults, and an admin ONLY if ADMIN_EMAIL /
    ADMIN_PASSWORD are set in the environment (SQLite)."""
    from werkzeug.security import generate_password_hash

    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE role = 'admin'")
    if not cur.fetchone():
        admin = _admin_from_env()
        if admin:
            cur.execute('''
                INSERT INTO users (employee_id, name, email, phone, department, password_hash, role, face_registered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', ('ADMIN001', admin["name"], admin["email"], '0000000000', 'Management',
                  generate_password_hash(admin["password"]), 'admin', 0))
            print(f"[Setup] Admin account created from ADMIN_EMAIL/ADMIN_PASSWORD env vars: {admin['email']}")
        else:
            print("[Setup] No admin account exists and ADMIN_EMAIL/ADMIN_PASSWORD are not set. "
                  "Run `python scripts/create_admin.py` to create one.")

    for key, value in _DEFAULT_SETTINGS.items():
        cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))


def _insert_defaults_pg(conn):
    """Insert settings defaults, and an admin ONLY if ADMIN_EMAIL /
    ADMIN_PASSWORD are set in the environment (PostgreSQL)."""
    from werkzeug.security import generate_password_hash

    result = conn.execute("SELECT id FROM users WHERE role = 'admin'").fetchone()
    if not result:
        admin = _admin_from_env()
        if admin:
            conn.execute('''
                INSERT INTO users (employee_id, name, email, phone, department, password_hash, role, face_registered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', ('ADMIN001', admin["name"], admin["email"], '0000000000', 'Management',
                  generate_password_hash(admin["password"]), 'admin', 0))
            print(f"[Setup] Admin account created from ADMIN_EMAIL/ADMIN_PASSWORD env vars: {admin['email']}")
        else:
            print("[Setup] No admin account exists and ADMIN_EMAIL/ADMIN_PASSWORD are not set. "
                  "Run `python scripts/create_admin.py` to create one.")

    for key, value in _DEFAULT_SETTINGS.items():
        conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT (key) DO NOTHING", (key, value))


def execute_query(conn, query, params=None):
    """Execute a query. The connection wrapper handles placeholder conversion."""
    return conn.execute(query, params) if params else conn.execute(query)


def fetchone(conn, query, params=None):
    """Fetch one row, returning dict-like object."""
    if params:
        return conn.execute(query, params).fetchone()
    return conn.execute(query).fetchone()


def fetchall(conn, query, params=None):
    """Fetch all rows, returning list of dict-like objects."""
    if params:
        return conn.execute(query, params).fetchall()
    return conn.execute(query).fetchall()


def get_setting(key, default=None):
    """Get a setting value from the settings table."""
    conn = get_db_connection()
    row = fetchone(conn, "SELECT value FROM settings WHERE key = ?", (key,))
    conn.close()
    return row['value'] if row else default


def set_setting(key, value):
    """Set a setting value in the settings table."""
    conn = get_db_connection()
    if Config.USE_POSTGRES:
        execute_query(conn, "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = ?", (key, str(value), str(value)))
    else:
        execute_query(conn, "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()
