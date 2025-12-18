from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import inference_router
from logger import get_logger, setup_logging

# Set up logging and get the logger
setup_logging()
logger = get_logger(__name__)

tags_metadata = [
    {
        "name": "inference",
        "description": "Endpoints related to document inference operations."
    },
    {
        "name": "test",
        "description": "Testing endpoints to verify connectivity and functionality."
    }
]


# Create app
app = FastAPI(title="Onboarding API",
              version="0.1",
              description="Obtain fields from documents through inference models.",
              openapi_tags=tags_metadata
              )

####################################################
# Added this to make it compatible with Angular calls
####################################################
# CORS configuration
####################################################

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)


# Declare router to access inference endpoint(s)
app.include_router(inference_router.router)

# Create middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code} for {request.method} {request.url}")
        return response
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the Inference API application.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the Inference API application")