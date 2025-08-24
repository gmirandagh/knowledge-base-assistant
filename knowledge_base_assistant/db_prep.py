from knowledge_base_assistant.db import init_db

if __name__ == "__main__":
    print("Initializing the database...")
    init_db()
    print("Database initialization complete.")