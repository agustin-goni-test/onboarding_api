import requests
import argparse
import os
from logger import setup_logging, get_logger
from utils import TimeMeasure

setup_logging()
logger = get_logger(__name__)
timer = TimeMeasure()

URL = "http://127.0.0.1:8000/onboarding/inference/infer"
FILE_PATH = "sources/CI1.pdf"
# FILE_PATH = "sources/carnet.jpg"
# FILE_PATH = "sources/image.jpg"

def send():
    '''
    Method to test sending a file and parameters through the inference endpoint
    '''
    # Measure overall time of operation
    id = timer.start_measurement()
    
    # Obtain arguments passed in the call
    parser = argparse.ArgumentParser(description="Call inference endpoint with a file.")
    parser.add_argument("--contact", type=str, nargs="+", help="Use the inference endpoint to obtain contact info.")
    parser.add_argument("--account", type=str, nargs="+", help="Use the inference endpoint to obtain contact info.")
    parser.add_argument("--commerce", type=str, nargs="+", help="Use the inference endpoint to obtain commerce info.")
    parser.add_argument("--participation", type=str, nargs="+", help="Use inference endpoint to obtain only the participation.")

    args = parser.parse_args()

    # Detemine the type of inference
    if args.contact:
        inference_type = "contact"
        filenames = args.contact
    elif args.account:
        inference_type = "account"
        filenames = args.account
    elif args.commerce:
        inference_type = "commerce"
        filenames = args.commerce
    elif args.participation:
        inference_type = "participation"
        filenames = args.participation
    else:
        logger.error("Inference type not recognized")
        return
    
    files_payloads = []

    # Iterate through all the filenames provided
    for filename in filenames:
        file_path = os.path.join("sources", filename)

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(file_path)
        
        logger.info(f"FILE TO PROCESS from arguments: {file_path}")

        files_payloads.append(
            ("files",
                (
                    filename,
                    open(file_path, "rb"),
                    "application/octet-stream"
                )
            )
        )


    # Echo the calling parameters
    logger.info(f"INFERENCE TYPE from arguments: {inference_type}")
    # logger.info(f"FILE TO PROCESS from arguments: {file_path}"


    # Data for the call, includes parameters
    data = {
        "inference_type": inference_type,
        "user_id": "1"
    }

    # Log the data for the call
    logger.info(f"Data for the call: {data}")

    # Add the file in question
    # with open(file_path, "rb") as f:
    #     files = {
    #         "files": (file_path, f, "application/octet-stream")
    #     }

    #     # Log the file path
    #     logger.info(f"File path: {file_path}")

    #     # Call service
    #     try:
    #         response = requests.post(URL, data=data, files=files)
    #         logger.info("Call to inference endpoint successful.")
        
    #     except Exception as e:
    #         logger.error(f"Call to inference endpoint failed: {str(e)}")

    try:
        response = requests.post(URL, data=data, files=files_payloads)
    except Exception as e:
        logger.error(f"Call to inference endpoint failed: {str(e)}")
        return

    print("\nStatus:", response.status_code)
    print("\nResponse:", response.text)
    print("\n")

    message = timer.calculate_time_elapsed(id)
    logger.info(message)


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

