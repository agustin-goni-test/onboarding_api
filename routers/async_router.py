from uuid import uuid4
from fastapi import APIRouter, Form, File, UploadFile

from tasks.inference_worker import run_inference_job
from db.db import create_job
from logger import setup_logging, get_logger
from files.files import save_files

setup_logging()
logger = get_logger(__name__)   

router = APIRouter(prefix="/onboarding/async-inference")

@router.post("/async-infer", status_code=202)
async def async_inference(
    inference_type: str = Form(...),
    user_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    '''
    Endpoint to submit an asynchronous inference job.
    '''

    # Get a new job id
    job_id = uuid4()

    # Create the job in the database
    create_job(
        job_id=str(job_id),
        inference_type=inference_type,
        user_id=user_id,
        status="PENDING"
    )

    save_files(str(job_id), files)

    run_inference_job.delay(job_id)

    return {
        "job_id": str(job_id),
        "status": "PENDING",
        "message": "Job submitted successfully. Processing will occur asynchronously."
    }
