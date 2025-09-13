import sqlite3
from typing import List, Tuple


# ===============================
# New Class for QA History (SQLite)
# ===============================
class QAHistory:
    def __init__(self, username: str, db_path: str = "qa_history.db"):
        self.username = username
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite DB and table"""
        conn = sqlite3.connect(self.db_path)
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
        conn.close()

    def save(self, question: str, answer: str):
        """Save a single Q&A pair to DB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO qa_history (username, question, answer) VALUES (?, ?, ?)",
            (self.username, question, answer),
        )
        conn.commit()
        conn.close()

    def load(self) -> List[Tuple[str, str]]:
        """Load all Q&A history for this user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question, answer FROM qa_history WHERE username=? ORDER BY created_at ASC",
            (self.username,),
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    def clear(self):
        """Clear all Q&A history for this user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM qa_history WHERE username=?", (self.username,))
        conn.commit()
        conn.close()
