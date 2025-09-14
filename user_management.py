import sqlite3
from datetime import datetime
import bcrypt
import os
from typing import Dict, List, Tuple, Any

SQLITE_DB_FILE = os.getenv("SQLITE_DB_FILE")

class UserDao:
    """Data Access Object for user management."""

    def __init__(self, db_file: str = None):
        """
        Initializes the UserDao.

        Args:
            db_file (str, optional): The path to the SQLite database file. 
                                     If None, it uses the SQLITE_DB_FILE 
                                     environment variable. Defaults to None.
        """
        db_path = db_file or SQLITE_DB_FILE
        if not db_path:
            raise ValueError(
                "Database file path cannot be empty. "
                "Please set SQLITE_DB_FILE environment variable or pass db_file."
            )
        self.db_file = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a new database connection."""
        return sqlite3.connect(self.db_file)

    def _init_db(self):
        """Initialize SQLite DB and users table if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.commit()

    def _hash_password(self, password: str) -> str:
        """Hashes a password using bcrypt."""
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verifies a password against a hashed version."""
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    def add_user(self, username: str, password: str, role: str = "user"):
        """Adds a new user to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            hashed_pw = self._hash_password(password)
            try:
                cursor.execute(
                    "INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, ?)",
                    (username, hashed_pw, role, datetime.utcnow().isoformat()),
                )
                conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"User '{username}' already exists.") from e

    def get_users(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves all users for streamlit-authenticator.
        The passwords are a HASHED version of the real password.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # The authenticator needs name and password. Role is custom.
            cursor.execute("SELECT username, password, role FROM users")
            data = cursor.fetchall()

        users = {}
        for u, p, r in data:
            users[u] = {"name": u, "password": p, "role": r}
        return users

    def list_users(self) -> List[Tuple[str, str, str]]:
        """
        Lists all users with their role and creation date for display.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username, role, created_at FROM users")
            rows = cursor.fetchall()
        return rows

