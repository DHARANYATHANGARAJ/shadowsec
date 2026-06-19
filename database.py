"""
database.py
------------
SQLite persistence layer for ShadowSec.

Tables
------
users        : accounts (admin + user roles)
otp_codes    : one-time-passcodes for login / password reset
datasets     : metadata about every uploaded dataset
predictions  : one row per analyzed dataset (attack type, confidence, risk...)
reports      : generated PDF incident reports
settings     : admin-configurable JSON blobs (risk thresholds, OTP policy...)
audit_log    : lightweight activity trail shown on the admin analytics page

All functions open a short-lived connection per call (SQLite + Streamlit's
rerun model makes long-lived connections risky across threads).
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import config
import security


@contextmanager
def get_connection():
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables (if they don't exist yet) and seed a default admin."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS otp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                code TEXT NOT NULL,
                purpose TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                attempts_used INTEGER NOT NULL DEFAULT 0,
                is_used INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                stored_path TEXT,
                rows INTEGER,
                cols INTEGER,
                missing_values INTEGER,
                duplicate_values INTEGER,
                class_distribution TEXT,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                user_id INTEGER NOT NULL,
                attack_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_category TEXT NOT NULL,
                top_features TEXT,
                explanation_text TEXT,
                model_used TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                pdf_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                category TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

    _seed_defaults()


def _seed_defaults():
    """Seed a default admin account and default settings on first run."""
    if get_user_by_username(config.SEED_ADMIN_USERNAME) is None:
        create_user(
            full_name="System Administrator",
            email=config.SEED_ADMIN_EMAIL,
            username=config.SEED_ADMIN_USERNAME,
            password=config.SEED_ADMIN_PASSWORD,
            role="admin",
        )

    if get_setting("password_policy") is None:
        set_setting("password_policy", config.DEFAULT_PASSWORD_POLICY)
    if get_setting("otp_settings") is None:
        set_setting("otp_settings", config.DEFAULT_OTP_SETTINGS)
    if get_setting("risk_bands") is None:
        set_setting("risk_bands", config.DEFAULT_RISK_BANDS)
    if get_setting("attack_severity") is None:
        set_setting("attack_severity", config.DEFAULT_ATTACK_SEVERITY)


# ---------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------
def create_user(full_name, email, username, password, role="user"):
    password_hash, salt = security.hash_password(password)
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO users (full_name, email, username, password_hash,
                                   password_salt, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                full_name,
                email.lower().strip(),
                username.strip(),
                password_hash,
                salt,
                role,
                datetime.utcnow().isoformat(),
            ),
        )
    return get_user_by_username(username)


def get_user_by_username(username):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_email(email):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def list_users():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, full_name, email, username, role, is_active, created_at "
            "FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_user(user_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def set_user_active(user_id, is_active: bool):
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?", (int(is_active), user_id)
        )


def set_user_role(user_id, role):
    with get_connection() as conn:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))


def update_password(user_id, new_password):
    password_hash, salt = security.hash_password(new_password)
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, password_salt = ? WHERE id = ?",
            (password_hash, salt, user_id),
        )


def authenticate(username, password):
    """Return user dict if credentials are valid, else None."""
    user = get_user_by_username(username)
    if not user or not user["is_active"]:
        return None
    if security.verify_password(password, user["password_hash"], user["password_salt"]):
        return user
    return None


# ---------------------------------------------------------------------
# OTP
# ---------------------------------------------------------------------
def create_otp(user_id, code, purpose, validity_minutes):
    now = datetime.utcnow()
    expires = now + timedelta(minutes=validity_minutes)
    with get_connection() as conn:
        # Invalidate any previous unused OTPs of the same purpose
        conn.execute(
            "UPDATE otp_codes SET is_used = 1 WHERE user_id = ? AND purpose = ? AND is_used = 0",
            (user_id, purpose),
        )
        conn.execute(
            """INSERT INTO otp_codes (user_id, code, purpose, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, code, purpose, now.isoformat(), expires.isoformat()),
        )


def get_active_otp(user_id, purpose):
    with get_connection() as conn:
        row = conn.execute(
            """SELECT * FROM otp_codes WHERE user_id = ? AND purpose = ?
               AND is_used = 0 ORDER BY id DESC LIMIT 1""",
            (user_id, purpose),
        ).fetchone()
        return dict(row) if row else None


def increment_otp_attempts(otp_id):
    with get_connection() as conn:
        conn.execute(
            "UPDATE otp_codes SET attempts_used = attempts_used + 1 WHERE id = ?",
            (otp_id,),
        )


def mark_otp_used(otp_id):
    with get_connection() as conn:
        conn.execute("UPDATE otp_codes SET is_used = 1 WHERE id = ?", (otp_id,))


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
def save_dataset(user_id, filename, stored_path, rows, cols, missing, duplicates, class_dist):
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO datasets (user_id, filename, stored_path, rows, cols,
                                      missing_values, duplicate_values,
                                      class_distribution, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                filename,
                stored_path,
                rows,
                cols,
                missing,
                duplicates,
                json.dumps(class_dist),
                datetime.utcnow().isoformat(),
            ),
        )
        return cur.lastrowid


def list_datasets(user_id=None):
    with get_connection() as conn:
        if user_id is None:
            rows = conn.execute(
                """SELECT d.*, u.username FROM datasets d
                   JOIN users u ON u.id = d.user_id ORDER BY d.uploaded_at DESC"""
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------
def save_prediction(user_id, dataset_id, attack_type, confidence, severity,
                     risk_score, risk_category, top_features, explanation_text,
                     model_used):
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO predictions (dataset_id, user_id, attack_type, confidence,
                                         severity, risk_score, risk_category,
                                         top_features, explanation_text, model_used,
                                         created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                dataset_id,
                user_id,
                attack_type,
                confidence,
                severity,
                risk_score,
                risk_category,
                json.dumps(top_features),
                explanation_text,
                model_used,
                datetime.utcnow().isoformat(),
            ),
        )
        return cur.lastrowid


def get_prediction(prediction_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        return dict(row) if row else None


def list_predictions(user_id=None):
    with get_connection() as conn:
        if user_id is None:
            rows = conn.execute(
                """SELECT p.*, u.username FROM predictions p
                   JOIN users u ON u.id = p.user_id ORDER BY p.created_at DESC"""
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------
def save_report(prediction_id, user_id, pdf_path):
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO reports (prediction_id, user_id, pdf_path, created_at)
               VALUES (?, ?, ?, ?)""",
            (prediction_id, user_id, pdf_path, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def list_reports(user_id=None):
    with get_connection() as conn:
        if user_id is None:
            rows = conn.execute(
                """SELECT r.*, u.username, p.attack_type, p.risk_score, p.risk_category
                   FROM reports r
                   JOIN users u ON u.id = r.user_id
                   JOIN predictions p ON p.id = r.prediction_id
                   ORDER BY r.created_at DESC"""
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT r.*, p.attack_type, p.risk_score, p.risk_category
                   FROM reports r
                   JOIN predictions p ON p.id = r.prediction_id
                   WHERE r.user_id = ? ORDER BY r.created_at DESC""",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Settings (generic JSON key/value store)
# ---------------------------------------------------------------------
def get_setting(category):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT data FROM settings WHERE category = ?", (category,)
        ).fetchone()
        return json.loads(row["data"]) if row else None


def set_setting(category, data: dict):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO settings (category, data) VALUES (?, ?)
               ON CONFLICT(category) DO UPDATE SET data = excluded.data""",
            (category, json.dumps(data)),
        )


# ---------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------
def log_action(user_id, action, details=""):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit_log (user_id, action, details, created_at) VALUES (?, ?, ?, ?)",
            (user_id, action, details, datetime.utcnow().isoformat()),
        )


def recent_activity(limit=20):
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT a.*, u.username FROM audit_log a
               LEFT JOIN users u ON u.id = a.user_id
               ORDER BY a.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Aggregate stats (used by landing page + analytics dashboard)
# ---------------------------------------------------------------------
def get_platform_stats():
    with get_connection() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_datasets = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        total_attacks = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE attack_type != 'Benign'"
        ).fetchone()[0]
        critical_alerts = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE risk_category = 'Critical'"
        ).fetchone()[0]
        avg_risk = conn.execute("SELECT AVG(risk_score) FROM predictions").fetchone()[0]
        total_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        return {
            "total_users": total_users,
            "total_datasets": total_datasets,
            "total_predictions": total_predictions,
            "total_attacks": total_attacks,
            "critical_alerts": critical_alerts,
            "avg_risk_score": round(avg_risk, 1) if avg_risk else 0.0,
            "total_reports": total_reports,
        }
