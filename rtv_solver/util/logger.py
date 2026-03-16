import logging
import json
import datetime

from logging import FileHandler, StreamHandler, Formatter

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
    

def setup_loggers(output_dir: str):
    """
    BASIC_LOGGER: writes stream output to terminal and a .txt log file
    
    DATA_LOGGER: writes structured data into a .JSONL file
    """
    LOG_FORMAT = ("%(asctime)s [%(levelname)s] %(message)s")
    LOG_LEVEL = logging.INFO

    basic_logger = logging.getLogger(BASIC_LOGGER)
    basic_logger.setLevel(LOG_LEVEL)
    basic_logger.propagate = False
    # basic_logger.propagate = False
    basic_logger_fileHandler = FileHandler(output_dir / "main.log")
    basic_logger_fileHandler.setFormatter(Formatter(LOG_FORMAT))
    basic_logger.addHandler(basic_logger_fileHandler)
    
    basic_logger_streamHandler = StreamHandler()
    basic_logger_streamHandler.setFormatter(Formatter(LOG_FORMAT))
    basic_logger.addHandler(basic_logger_streamHandler)

    # JSON logger that collects data during the runtime that we can later use to analyse behavior, trip generation, which trips were available, which requests were active in that moment in addition to the manifest, with timestamp of current_time of iteration, so we can rerun the behavior visibly in geoJson and possbily folium.plugins.Timeline and understand it better on a small scale.
    behavior_logger = logging.getLogger(DATA_LOGGER)
    behavior_logger.setLevel(LOG_LEVEL)
    behavior_logger.propagate = False
    
    behavior_logger.handlers.clear()
    json_file_handler = logging.FileHandler(output_dir / "assignment_data.jsonl")
    json_file_handler.setFormatter(JsonFormatter())
    behavior_logger.addHandler(json_file_handler)

    # information that should be stored here is the timestamp, assignment history, active requests, trip generation (possibly advanced and a lot of data), reassignments between vehicles

    # turn off logging by 'requests' package
    logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)