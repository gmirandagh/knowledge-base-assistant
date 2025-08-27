import os
import psycopg2
from psycopg2.extras import DictCursor, Json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Configuration
RUN_TIMEZONE_CHECK = os.getenv('RUN_TIMEZONE_CHECK', '1') == '1'
TZ_INFO = os.getenv("TZ", "Europe/Berlin")
tz = ZoneInfo(TZ_INFO)


def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "knowledge_base_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


def init_db():
    """Initializes the database schema by creating tables with enhanced monitoring fields."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name IN ('conversations', 'feedback')
            """)
            table_count = cur.fetchone()[0]
            
            if table_count > 0:
                cur.execute("SELECT COUNT(*) FROM conversations")
                conversation_count = cur.fetchone()[0]
                
                if conversation_count > 0:
                    print(f"✅ Database already initialized with {conversation_count} conversations. Skipping destructive initialization.")
                    return
                else:
                    print("📊 Tables exist but are empty. Proceeding with safe initialization...")
            
            print("🏗️ Creating database tables...")
            
            if table_count > 0:
                print("⚠️ Dropping existing empty tables...")
                cur.execute("DROP TABLE IF EXISTS feedback;")
                cur.execute("DROP TABLE IF EXISTS conversations;")

            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT,
                    context JSONB,
                    model_used TEXT,
                    response_time FLOAT,
                    relevance TEXT,
                    relevance_explanation TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    eval_prompt_tokens INTEGER,
                    eval_completion_tokens INTEGER,
                    eval_total_tokens INTEGER,
                    openai_cost FLOAT,
                    user_language TEXT DEFAULT 'en',
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                    feedback INTEGER NOT NULL CHECK (feedback IN (-1, 1)),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_relevance ON conversations(relevance);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_conversation_id ON feedback(conversation_id);")
            
        conn.commit()
        print("✅ Database tables created successfully with monitoring enhancements.")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def save_conversation(conversation_id, question, answer, context, metrics=None, user_language='en', timestamp=None):
    """
    Saves a new conversation to the database with optional monitoring metrics.
    
    Args:
        conversation_id: Unique identifier for the conversation
        question: The user's question
        answer: The generated answer
        context: The context used for generating the answer
        metrics: Optional dictionary containing monitoring metrics
        user_language: Language of the conversation
        timestamp: Optional timestamp (defaults to current time)
    """
    if timestamp is None:
        timestamp = datetime.now(tz)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if metrics:
                # Save with full metrics
                cur.execute(
                    """
                    INSERT INTO conversations 
                    (id, question, answer, context, model_used, response_time, relevance, 
                    relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                    eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, 
                    openai_cost, user_language, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        question,
                        answer,
                        Json(context),
                        metrics.get("model_used"),
                        metrics.get("processing_time_seconds"),
                        metrics.get("relevance_evaluation", {}).get("Relevance"),
                        metrics.get("relevance_evaluation", {}).get("Explanation"),
                        metrics.get("total_tokens", {}).get("prompt_tokens"),
                        metrics.get("total_tokens", {}).get("completion_tokens"),
                        metrics.get("total_tokens", {}).get("total_tokens"),
                        metrics.get("relevance_evaluation", {}).get("eval_prompt_tokens", 0),
                        metrics.get("relevance_evaluation", {}).get("eval_completion_tokens", 0),
                        metrics.get("relevance_evaluation", {}).get("eval_total_tokens", 0),
                        metrics.get("total_cost_usd"),
                        user_language,
                        timestamp
                    ),
                )
            else:
                # Save basic conversation (backward compatible)
                cur.execute(
                    """
                    INSERT INTO conversations (id, question, answer, context, user_language, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        question,
                        answer,
                        Json(context),
                        user_language,
                        timestamp
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def save_feedback(conversation_id, feedback_value, timestamp=None):
    """
    Saves user feedback to the database.
    
    Args:
        conversation_id: ID of the conversation being rated
        feedback_value: 1 for thumbs up, -1 for thumbs down
        timestamp: Optional timestamp (defaults to current time)
    """
    if timestamp is None:
        timestamp = datetime.now(tz)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, %s)",
                (conversation_id, feedback_value, timestamp),
            )
        conn.commit()
    finally:
        conn.close()


def get_recent_conversations(limit=5, relevance=None, user_language=None):
    """
    Retrieves recent conversations with optional filtering.
    
    Args:
        limit: Maximum number of conversations to return
        relevance: Filter by relevance level (e.g., 'RELEVANT', 'PARTLY_RELEVANT', 'NON_RELEVANT')
        user_language: Filter by user language
        
    Returns:
        List of conversation dictionaries with feedback information
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = """
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
            """
            conditions = []
            params = []
            
            if relevance:
                conditions.append("c.relevance = %s")
                params.append(relevance)
            
            if user_language:
                conditions.append("c.user_language = %s")
                params.append(user_language)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY c.timestamp DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()


def get_feedback_stats():
    """
    Retrieves feedback statistics.
    
    Returns:
        Dictionary with thumbs_up and thumbs_down counts
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as thumbs_down
                FROM feedback
            """)
            result = cur.fetchone()
            return {
                'thumbs_up': result['thumbs_up'] or 0,
                'thumbs_down': result['thumbs_down'] or 0
            }
    finally:
        conn.close()


def get_conversation_stats(days=7):
    """
    Retrieves conversation statistics for monitoring dashboard.
    
    Args:
        days: Number of days to include in statistics
        
    Returns:
        Dictionary with various statistics
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    AVG(response_time) as avg_response_time,
                    SUM(openai_cost) as total_cost,
                    AVG(openai_cost) as avg_cost_per_conversation,
                    SUM(total_tokens) as total_tokens_used,
                    COUNT(CASE WHEN relevance = 'RELEVANT' THEN 1 END) as relevant_count,
                    COUNT(CASE WHEN relevance = 'PARTLY_RELEVANT' THEN 1 END) as partly_relevant_count,
                    COUNT(CASE WHEN relevance = 'NON_RELEVANT' THEN 1 END) as non_relevant_count
                FROM conversations 
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND response_time IS NOT NULL
            """, (days,))
            
            stats = cur.fetchone()
            
            # Get language distribution
            cur.execute("""
                SELECT user_language, COUNT(*) as count
                FROM conversations 
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                GROUP BY user_language
                ORDER BY count DESC
            """, (days,))
            
            language_stats = cur.fetchall()
            
            return {
                'total_conversations': stats['total_conversations'] or 0,
                'avg_response_time': float(stats['avg_response_time'] or 0),
                'total_cost': float(stats['total_cost'] or 0),
                'avg_cost_per_conversation': float(stats['avg_cost_per_conversation'] or 0),
                'total_tokens_used': stats['total_tokens_used'] or 0,
                'relevance_distribution': {
                    'relevant': stats['relevant_count'] or 0,
                    'partly_relevant': stats['partly_relevant_count'] or 0,
                    'non_relevant': stats['non_relevant_count'] or 0
                },
                'language_distribution': [dict(row) for row in language_stats]
            }
    finally:
        conn.close()


def get_conversation_by_id(conversation_id):
    """
    Retrieves a specific conversation by ID.
    
    Args:
        conversation_id: The conversation ID to retrieve
        
    Returns:
        Conversation dictionary or None if not found
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
                WHERE c.id = %s
            """, (conversation_id,))
            return cur.fetchone()
    finally:
        conn.close()


def check_timezone():
    """
    Checks and displays timezone information for debugging.
    Useful for ensuring proper timestamp handling across different environments.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW timezone;")
            db_timezone = cur.fetchone()[0]
            print(f"Database timezone: {db_timezone}")

            cur.execute("SELECT current_timestamp;")
            db_time_utc = cur.fetchone()[0]
            print(f"Database current time (UTC): {db_time_utc}")

            db_time_local = db_time_utc.astimezone(tz)
            print(f"Database current time ({TZ_INFO}): {db_time_local}")

            py_time = datetime.now(tz)
            print(f"Python current time: {py_time}")

            # Test insertion with current implementation
            test_id = f"timezone_test_{int(datetime.now().timestamp())}"
            cur.execute("""
                INSERT INTO conversations 
                (id, question, answer, user_language, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING timestamp;
            """, (test_id, 'test question', 'test answer', 'en', py_time))

            inserted_time = cur.fetchone()[0]
            print(f"Inserted time (UTC): {inserted_time}")
            print(f"Inserted time ({TZ_INFO}): {inserted_time.astimezone(tz)}")

            cur.execute("SELECT timestamp FROM conversations WHERE id = %s;", (test_id,))
            selected_time = cur.fetchone()[0]
            print(f"Selected time (UTC): {selected_time}")
            print(f"Selected time ({TZ_INFO}): {selected_time.astimezone(tz)}")

            # Clean up the test entry
            cur.execute("DELETE FROM conversations WHERE id = %s;", (test_id,))
            conn.commit()
            
            print("✅ Timezone check completed successfully!")
            
    except Exception as e:
        print(f"❌ An error occurred during timezone check: {e}")
        conn.rollback()
    finally:
        conn.close()


def migrate_existing_data():
    """
    Helper function to migrate existing conversations to new schema.
    Run this once when upgrading from the basic schema.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'conversations' AND column_name IN 
                ('model_used', 'response_time', 'relevance')
            """)
            
            existing_columns = [row[0] for row in cur.fetchall()]
            
            if len(existing_columns) < 3:
                print("Adding new monitoring columns to existing conversations table...")
                
                new_columns = [
                    "ADD COLUMN IF NOT EXISTS model_used TEXT",
                    "ADD COLUMN IF NOT EXISTS response_time FLOAT",
                    "ADD COLUMN IF NOT EXISTS relevance TEXT",
                    "ADD COLUMN IF NOT EXISTS relevance_explanation TEXT",
                    "ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER",
                    "ADD COLUMN IF NOT EXISTS completion_tokens INTEGER", 
                    "ADD COLUMN IF NOT EXISTS total_tokens INTEGER",
                    "ADD COLUMN IF NOT EXISTS eval_prompt_tokens INTEGER",
                    "ADD COLUMN IF NOT EXISTS eval_completion_tokens INTEGER",
                    "ADD COLUMN IF NOT EXISTS eval_total_tokens INTEGER",
                    "ADD COLUMN IF NOT EXISTS openai_cost FLOAT",
                    "ADD COLUMN IF NOT EXISTS user_language TEXT DEFAULT 'en'"
                ]
                
                for column_def in new_columns:
                    cur.execute(f"ALTER TABLE conversations {column_def}")
                
                cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_relevance ON conversations(relevance);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_conversation_id ON feedback(conversation_id);")
                
                conn.commit()
                print("✅ Migration completed successfully!")
            else:
                print("✅ Database schema is already up to date.")
                
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


# Run timezone check if enabled
if RUN_TIMEZONE_CHECK:
    print("🕐 Running timezone check...")
    check_timezone()


# Optional: Run migration check on import
if os.getenv('AUTO_MIGRATE', '0') == '1':
    print("🔄 Running automatic migration check...")
    try:
        migrate_existing_data()
    except Exception as e:
        print(f"⚠️ Auto-migration failed: {e}")
        print("Please run migrate_existing_data() manually if needed.")