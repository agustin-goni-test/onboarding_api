from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from logger import setup_logging, get_logger
from workflow.workflow_agent import InferenceAgent, InferenceState
import json

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


@router.post("/infer",
             summary="Endpoint that carries out an inference to obtain document data.",
             description="Receives files and inference type to obtain relevant fields from the document(s).",
             responses={
                    200: {
                        "description": "Inference completed successfully.",
                        "content": {
                            "application/json": {
                                "example": {
                                    "inference_type": "contact",
                                    "user_id": "12345678-9",
                                    "filename": ["document1.pdf", "document2.jpg"],
                                    "file_count": 2,
                                    "data": [
                                        {
                                            "field": "name",
                                            "found": True,
                                            "found_by": ["model_a", "llm_b"],
                                            "values": ["John Doe", "J. Doe"],
                                            "explanations": [
                                                "Extracted from the header using model_a.",
                                                "Inferred from context using llm_b."
                                            ],
                                            "probable_value": "John Doe",
                                            "confidences": [95, 85],
                                            "low_confidence": False,
                                            "found_multiple": True
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    400: {
                        "description": "Bad Request - No files uploaded."},
                    500: {
                        "description": "Internal Server Error - An error occurred during inference."}
    
                }
             )
async def inference_manager(
    inference_type: str = Form(...),
    user_id: str = Form(...),
    # file: UploadFile = File(...)
    files: list[UploadFile] = File(...)
):
    '''
    Receives the following (tentative):
    - an inference_type which instructs what fields to search for
    - user_id, probably the relevant RUT
    - file: uploaded file
    '''

    if not files:
        logger.error("No files uploaded for inference.")
        return JSONResponse(
            status_code=400,
            content={"error": "No files uploaded for inference."}
        )
    
    # first_file = files[0]

    file_payloads = []
    filenames = []

    logger.info(f"Received inference call with type: {inference_type}, user: {user_id} and a list of files")

    # Obtain file as byte stream from the UploadFile object of FastAPI
    # file_bytes = await first_file.read()
    # file_name = first_file.filename

    # Append all the files in the request
    for f in files:
        file_bytes = await f.read()
        file_payloads.append( (file_bytes, f.filename) )
        filenames.append(f.filename)
        logger.info(f"File to use in inference: {f.filename}")

    logger.info(f"Total files received for inference: {len(file_payloads)}")
    

    # Create agent and pass the file
    agent = InferenceAgent()
    agent.feed_files_to_agent(file_payloads)
    agent.set_type_of_inference(inference_type)
    logger.info("Inference agent created...")

    # Set initial (empty) state
    initial_state = InferenceState()

    final_state = agent.capture_data(initial_state=initial_state)
    # data_list = json.loads(final_state["response_format"])

    json_data = json.dumps(
        final_state["response_format"],
        ensure_ascii=False
    )

    return JSONResponse(
        content={
            "inference_type": inference_type,
            "user_id": user_id,
            "filename": filenames,
            "file_count": len(filenames),
            "data": final_state["response_format"]
        }
    )