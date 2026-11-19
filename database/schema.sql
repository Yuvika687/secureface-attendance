CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    employee_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    department TEXT,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'employee' CHECK(role IN ('admin', 'employee')),
    profile_photo TEXT,
    face_encoding BYTEA,
    face_encodings TEXT,
    face_embedding TEXT,
    face_registered INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
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
-- frame -- app.py only inserts when the decision differs from the last one
-- logged for that session.
CREATE TABLE IF NOT EXISTS liveness_events (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('LIVE', 'SPOOF', 'UNCERTAIN')),
    p_spoof REAL NOT NULL,
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One row per scan-pipeline STAGE reached per request (deepest stage only,
-- deduped against the previous frame's stage for the continuous states --
-- see app.py's _log_scan_event()). For reviewing a test session after the
-- fact: what happened, in order, with timestamps.
CREATE TABLE IF NOT EXISTS scan_events (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    stage TEXT NOT NULL CHECK(stage IN (
        'NO_FACE', 'MULTI_FACE', 'FACE_MOVED', 'SPOOF', 'UNCERTAIN',
        'RECOGNIZED', 'UNRECOGNIZED', 'CHALLENGE_PASSED',
        'ATTENDANCE_MARKED', 'ALREADY_MARKED'
    )),
    user_id INTEGER,
    detail TEXT,
    test_label TEXT,
    p_spoof_frame REAL,
    p_spoof_avg REAL,
    iou REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Existing Postgres databases: add the per-frame score columns.
ALTER TABLE scan_events ADD COLUMN IF NOT EXISTS p_spoof_frame REAL;
ALTER TABLE scan_events ADD COLUMN IF NOT EXISTS p_spoof_avg REAL;
ALTER TABLE scan_events ADD COLUMN IF NOT EXISTS iou REAL;

-- Admin Test Mode: one row per camera frame, labeled with what was shown to
-- the camera. Never linked to attendance.
CREATE TABLE IF NOT EXISTS test_mode_results (
    id SERIAL PRIMARY KEY,
    test_session TEXT NOT NULL,
    category TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('LIVE', 'SPOOF', 'UNCERTAIN', 'NO_FACE', 'MULTI_FACE')),
    p_spoof_frame REAL,
    p_spoof_avg REAL,
    admin_user_id INTEGER,
    person_id TEXT,
    glasses TEXT,
    lighting TEXT,
    collection_session INTEGER,
    crop_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Existing Postgres databases: collection-mode columns (Phase 3).
ALTER TABLE test_mode_results ADD COLUMN IF NOT EXISTS person_id TEXT;
ALTER TABLE test_mode_results ADD COLUMN IF NOT EXISTS glasses TEXT;
ALTER TABLE test_mode_results ADD COLUMN IF NOT EXISTS lighting TEXT;
ALTER TABLE test_mode_results ADD COLUMN IF NOT EXISTS collection_session INTEGER;
ALTER TABLE test_mode_results ADD COLUMN IF NOT EXISTS crop_path TEXT;

-- Browsers an admin approved as the attendance scanner. The device holds a
-- random token in a cookie; only its SHA-256 hash is stored here.
CREATE TABLE IF NOT EXISTS scanner_devices (
    id SERIAL PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    label TEXT,
    approved_by INTEGER,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP
);

-- Audit trail for admin actions (wipe_model, scanner approve/revoke):
-- who did it and when.
CREATE TABLE IF NOT EXISTS admin_audit_log (
    id SERIAL PRIMARY KEY,
    admin_user_id INTEGER NOT NULL,
    admin_email TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    ip_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes for 500+ employee scale
-- The attendance lookup (user_id + date) is the most frequent query in the system
CREATE INDEX IF NOT EXISTS idx_attendance_user_date ON attendance(user_id, date);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_department ON users(department);
CREATE INDEX IF NOT EXISTS idx_liveness_events_created ON liveness_events(created_at);
CREATE INDEX IF NOT EXISTS idx_scan_events_created ON scan_events(created_at);
