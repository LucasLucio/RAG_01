import sqlite3
import json
import os
import uuid

DB_FILE = "log_questions.db"


# -----------------------------------------------------
# 1) Função para criar banco e tabela (se não existirem)
# -----------------------------------------------------
def start_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            question TEXT,
            datetime_start TEXT,
            datetime_end TEXT,
            data_json TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Banco e tabela prontos.")


# -----------------------------------------------------
# 2) Inserir log com ID UUID
# -----------------------------------------------------
def insert_log(id, data: dict):
    record_id = id
    question = data.get("question")
    datetime_start = data.get("datetime_start")
    datetime_end = data.get("datetime_end")
    data_json = json.dumps(data, ensure_ascii=False)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO logs (id, question, datetime_start, datetime_end, data_json)
        VALUES (?, ?, ?, ?, ?)
    """, (record_id, question, datetime_start, datetime_end, data_json))

    conn.commit()
    conn.close()

    print(f"Registro inserido com sucesso! ID = {record_id}")


# -----------------------------------------------------
# 3) Exemplo de uso
# -----------------------------------------------------
if __name__ == "__main__":
    start_database()

    exemplo_data = {
        "question": "O que é uma operation?",
        "datetime_start": "2025-11-12 20:13:08.651377",
        "datetime_end": "2025-11-12 20:13:42.030912",
        "default_return": False,
        "response": None,
        "executions_rag": [],
        "steps": ["judge", "router", "filter"]
    }

    insert_log(exemplo_data)