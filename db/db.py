import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID
from dotenv import load_dotenv

load_dotenv()

DBNAME = os.getenv("DBNAME")
DBUSER = os.getenv("DBUSER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

DATABASE_URL = f"postgresql+psycopg2://{DBUSER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"

@contextmanager
def get_db_connection():
    # Create connection
    conn = psycopg2.connect(DATABASE_URL)
    
    # Connect to database and yield connection
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_job(
        job_id: UUID,
        rut_comercio: str,
        user_id: str,
        inference_type: str,
        status: str="PENDING"
        ):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO jobs (job_id, rut_comercio, user_id, inference_type, status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (str(job_id), rut_comercio, user_id, inference_type, status)
            )


def get_job(job_id: UUID):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                """
                SELECT * FROM jobs WHERE job_id = %s
                """,
                (str(job_id),)
            )
            job = cursor.fetchone()
            return dict(job) if job else None
        

def mark_job_running(job_id: UUID):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE jobs
                SET status = %s, started_at = %s
                WHERE job_id = %s
                """,
                ("RUNNING", datetime.now(timezone.utc), str(job_id))
            )

def mark_job_done(job_id: UUID):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE jobs
                SET status = %s, completed_at = %s,
                error = NULL
                WHERE job_id = %s
                """,
                ("DONE", datetime.now(timezone.utc), str(job_id))
            )


def mark_job_failed(job_id: UUID, error_message: str):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE jobs
                SET status = %s, completed_at = %s,
                error = %s
                WHERE job_id = %s
                """,
                ("FAILED", datetime.now(timezone.utc), error_message, str(job_id))
            )
