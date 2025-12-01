import requests
from logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

URL = "http://127.0.0.1:8000/onboarding/inference/infer"
FILE_PATH = "sources/carnet_ficticio.pdf"

def send():
    '''
    Method to test sending a file and parameters through the inference endpoint
    '''
    
    # Data for the call, includes parameters
    data = {
        "inference_type": "contact",
        "user_id": "1"
    }

    # Log the data for the call
    logger.info(f"Data for the call: {data}")

    # Add the file in question
    with open(FILE_PATH, "rb") as f:
        files = {
            "file": (FILE_PATH, f, "application/octet-stream")
        }

        # Log the file path
        logger.info(f"File path: {FILE_PATH}")

        # Call service
        try:
            response = requests.post(URL, data=data, files=files)
            logger.info("Call to inference endpoint successful.")
        
        except Exception as e:
            logger.error(f"Call to inference endpoint failed: {str(e)}")

    print("Status:", response.status_code)
    print("Response:", response.text)


def send_with_visibility():
    '''
    Method to test sending a file and parameters through the inference endpoint.
    Includes the explicit preparation and echoing of the call details
    (to validate the input)
    '''
    
    # Data for the call, includes parameters
    data = {
        "inference_type": "contact",
        "user_id": "1"
    }

    # Add the file in question
    with open(FILE_PATH, "rb") as f:
        files = {
            "file": (FILE_PATH, f, "application/octet-stream")
        }

        # Create the raw request before sending it
        session = requests.Session()
        req = requests.Request("POST", URL, data=data, files=files)
        prepped = session.prepare_request(req)

        # Observe the exact byte stream to be sent
        raw_bytes = prepped.body

        # Output to screen
        print(raw_bytes)

        # Call service
        response = session.send(prepped)

    print("Status:", response.status_code)
    print("Response:", response.text)


if __name__ == "__main__":
    send()

