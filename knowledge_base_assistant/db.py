import os
import psycopg2
from psycopg2.extras import DictCursor, Json
from datetime import datetime

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "knowledge_base_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )

def init_db():
    """Initializes the database schema by creating tables."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Clean start
            cur.execute("DROP TABLE IF EXISTS feedback;")
            cur.execute("DROP TABLE IF EXISTS conversations;")

            # Conversations table
            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT,
                    context JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Feedback table
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                    feedback INTEGER NOT NULL CHECK (feedback IN (-1, 1)),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
        print("Database tables created successfully.")
    finally:
        conn.close()

def save_conversation(conversation_id, question, answer, context):
    """Saves a new conversation to the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (id, question, answer, context, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    question,
                    answer,
                    Json(context),
                    datetime.now()
                ),
            )
        conn.commit()
    finally:
        conn.close()

def save_feedback(conversation_id, feedback_value):
    """Saves user feedback to the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback) VALUES (%s, %s)",
                (conversation_id, feedback_value),
            )
        conn.commit()
    finally:
        conn.close()