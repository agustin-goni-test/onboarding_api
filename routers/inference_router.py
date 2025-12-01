from fastapi import APIRouter, File, Form, UploadFile
from logger import setup_logging, get_logger
from workflow.workflow_agent import InferenceAgent, InferenceState

# Use logging for call details
setup_logging()
logger = get_logger(__name__)

# Create router to add to main app
router = APIRouter(prefix="/onboarding/inference")

@router.get("/test", summary="Testing endpoint for the API.")
def test_endpoint():
    '''
    Simple endpoint to ensure connectivity
    '''

    return {"status": "Endpoint working properly"}


@router.post("/infer", summary="Endpoint that carries out an inference to obtain document data.")
async def inference_manager(
    inference_type: str = Form(...),
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    '''
    Receives the following (tentative):
    - an inference_type which instructs what fields to search for
    - user_id, probably the relevant RUT
    - file: uploaded file
    '''

    logger.info(f"Received inference call with type: {inference_type}, user: {user_id} and file: {file.filename}")

    # Obtain file as byte stream from the UploadFile object of FastAPI
    file_bytes = await file.read()

    # Create agent and pass the file
    agent = InferenceAgent()
    agent.feed_file_to_agent(file_bytes)
    logger.info("Inference agent created...")

    # Set initial (empty) state
    initial_state = InferenceState()

    final__state = agent.capture_data(initial_state=initial_state)

    return {
        "inference_type": inference_type,
        "user_id": user_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File and parameters correctly uploaded."
    }