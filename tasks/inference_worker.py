import httpx
from celery_app import celery_app
from db.db import (
    mark_job_running,
    mark_job_done,
    mark_job_failed,
    get_job
)

from files.files import load_files_for_http

@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 5}
)
def run_inference_job(self, job_id: str):
    '''
    Runs inference asynchronously using the infer/ endpoint.
    '''

    # Mark the job as running (timestamped)
    mark_job_running(job_id)

    try:
        # Get job, returns dict
        job =get_job(job_id)

        # Use client to call inference endpoint
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "http://localhost:8000/onboarding/inference/infer",
                data={
                    "inference_type": job["inference_type"],
                    "user_id": job["user_id"]
                },
                files=load_files_for_http(job_id)
            )

        # Obtain response and result
        response.raise_for_status()
        result = response.json()

        # Mark job as done (timestamped)
        mark_job_done(job_id)

    except Exception as e:
        mark_job_failed(job_id)
        raise
