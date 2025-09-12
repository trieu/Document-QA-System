import sqlite3
from datetime import datetime
import bcrypt


def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT,
            created_at TEXT
        )"""
    )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def add_user(username, password, role="user"):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed_pw = hash_password(password)
    c.execute(
        "INSERT INTO users VALUES (?, ?, ?, ?)",
        (username, hashed_pw, role, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_users():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username, password, role FROM users")
    data = c.fetchall()
    conn.close()
    return {u: {"password": p, "role": r} for u, p, r in data}


def list_users():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username, role, created_at FROM users")
    rows = c.fetchall()
    conn.close()
    return rows
