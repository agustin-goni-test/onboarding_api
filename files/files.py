import os
from typing import Iterable
from fastapi import UploadFile
from logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

async def save_files(job_id: str, files: list[UploadFile]) -> None:
    '''
    Save uploaded files to the specified path
    '''
    base = f"files/inference/{job_id}"
    os.makedirs(base, exist_ok=True)

    for file in files:
        contents = await file.read()
        with open(os.path.join(base, file.filename), "wb") as out:
            out.write(contents)
            logger.info(f"Saved file: {base}/{file.filename}")

def load_files_for_http(job_id: str) -> Iterable[tuple[str, tuple[str, bytes, str]]]:
    '''
    Load files from the specified path for HTTP transmission
    '''
    base = f"files/inference/{job_id}"
    file_payloads = []

    if not os.path.exists(base):
        logger.error(f"Directory does not exist: {base}")
        return file_payloads

    for filename in os.listdir(base):
        file_path = os.path.join(base, filename)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            file_payloads.append(
                ("files", (filename, file_bytes, "application/octet-stream"))
            )
            logger.info(f"Loaded file for HTTP: {file_path}")

    return file_payloads
    