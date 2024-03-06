import logging  #importing the logging module
import os 
from datetime import datetime  

#creating a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#creating a directory path for storing logs within the current working directory
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

#creating the logs directory if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,  #setting the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  #specifying the log message format
    level=logging.INFO,  
)
