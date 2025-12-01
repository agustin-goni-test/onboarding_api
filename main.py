from fastapi import FastAPI
from routers import inference_router
from logger import get_logger, setup_logging

# Set up logging and get the logger
setup_logging()
logger = get_logger(__name__)

# Create app
app = FastAPI(title="Onboarding API", version="0.1")

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