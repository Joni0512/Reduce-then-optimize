import logging
import json
import datetime

from logging import FileHandler, StreamHandler
from logging import Formatter

from pathlib import Path

from rtv_solver.structure.config import Config

# keys for the loggers
BASIC_LOGGER = "rtv_solver.basic"
DATA_LOGGER = "rtv_solver.data"

# these should not be used for the extra values
STANDARD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno",
    "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process"
}

class JsonFormatter(logging.Formatter):
    """
    collect extra information during code execution, so we do not have to hand content through all layers in order to collect the information
    
    For easier parsing for the analysis of the programme, the structure should always be the same.
    """
    def format(self, record):
        data = {
            "time": datetime.datetime.now().isoformat(),
            "level": record.levelname,
            # "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {}
        for key, value in record.__dict__.items():
            if key in STANDARD_ATTRS:
                continue
            extras[key] = value

        if extras:
            data["extra"] = extras

        return json.dumps(data)
def setup_logging(config: Config):
    ROOT_DIR = Path(__file__).resolve().parent
    output_dir = ROOT_DIR.parent / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'main.log'

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()]
        ) 
    
# classic logger that writes to the terminal and a log file 
def setup_loggers(config: Config, ):
    LOG_FORMAT = ("%(asctime)s [%(levelname)s] %(message)s")
    LOG_LEVEL = logging.INFO

    basic_logger = logging.getLogger(BASIC_LOGGER)
    basic_logger.setLevel(LOG_LEVEL)
    basic_logger.propagate = False
    # basic_logger.propagate = False
    # TODO how to import the output file here
    basic_logger_fileHandler = FileHandler(config.output_dir / "main.log")
    basic_logger_fileHandler.setFormatter(Formatter(LOG_FORMAT))
    basic_logger.addHandler(basic_logger_fileHandler)
    
    basic_logger_streamHandler = StreamHandler()
    basic_logger_streamHandler.setFormatter(Formatter(LOG_FORMAT))
    basic_logger.addHandler(basic_logger_streamHandler)

    # TODO Create a JSON logger that collects data during the runtime that we can later use to analyse behavior, trip generation, which trips were available, which requests were active in that moment in addition to the manifest, with timestamp of current_time of iteration, so we can rerun the behavior visibly in geoJson and possbily folium.plugins.Timeline and understand it better on a small scale.

    behavior_logger = logging.getLogger(DATA_LOGGER)
    behavior_logger.setLevel(LOG_LEVEL)
    behavior_logger.propagate = False
    
    behavior_logger.handlers.clear()
    json_file_handler = logging.FileHandler(config.output_dir / "behavior.jsonl")
    json_file_handler.setFormatter(JsonFormatter())
    behavior_logger.addHandler(json_file_handler)
    # TODO add parts required for JSON
    # information that should be stored here is the timestamp, assignment history, active requests, trip generation (advanced and a lot of data), reassignments between vehicles

    # turn off logging by 'requests' package
    logging.getLogger("requests.packages.urllib3").setLevel(logging.DEBUG) # TODO test

# TODO integrate 
def setup_directories(config: Config):
    # Define the directory name
    ROOT_DIR = Path(__file__).resolve().parent
    output_dir = ROOT_DIR.parent / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / "test"
    output_dir = Path("simulation_results")
    
    # Create it
    # parents=True: creates any missing folders in the path (like /a/b/c)
    # exist_ok=True: doesn't raise an error if the folder is already there
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir