import logging
import json
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)
    

def setup_logging():
    '''Setup application logging in JSON format'''

    # Set environment from configuration
    environment = os.getenv("ENVIRONMENT", "development")

    # Set logging level from configuration
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Set directory for log files from configuration
    log_dir = os.getenv("LOG_DIR", "logs")

    log_in_json = os.getenv("LOG_IN_JSON", "true").lower() == "true"
    log_in_console = os.getenv("LOG_IN_CONSOLE", "true").lower() == "true"

    # Create path if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler with plain text format
    text_file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    text_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Format file as plain text
    text_file_handler.setFormatter(text_formatter)

    # If JSON logging is enabled
    if log_in_json:
        # File habndler with JSON Format
        file_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, "app.json.log"),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )

        # Format file as JSON
        file_handler.setFormatter(JSONFormatter())

    # If logging in console is enabled
    if log_in_console:
        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(text_file_handler)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    

    # Set level for external libraries to WARNING to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    '''Get a logger instance by name'''
    return logging.getLogger(name)