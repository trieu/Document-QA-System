import os
import sqlite3
from typing import List, Tuple


SQLITE_DB_FILE=os.getenv("SQLITE_DB_FILE")

class UserChatDao:
    """Data Access Object for user Q&A chat history."""

    def __init__(self, db_file: str = None):
        """
        Initializes the UserChatDao.

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
        """Initialize SQLite DB and qa_history table if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save_qa(self, username: str, question: str, answer: str):
        """Save a single Q&A pair to the database for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO qa_history (username, question, answer) VALUES (?, ?, ?)",
                (username, question, answer),
            )
            conn.commit()

    def load_history(self, username: str) -> List[Tuple[str, str]]:
        """Load all Q&A history for a given user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT question, answer FROM qa_history WHERE username=? ORDER BY created_at ASC",
                (username,),
            )
            return cursor.fetchall()

    def clear_history(self, username: str):
        """Clear all Q&A history for a given user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM qa_history WHERE username=?", (username,))
            conn.commit()
