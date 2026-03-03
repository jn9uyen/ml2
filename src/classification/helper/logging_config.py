"""
Logging config
https://betterstack.com/community/guides/logging/python/python-logging-best-practices/
"""

import os
from cl.framework.utils import logging


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

folder = f"{parent_dir}/logs"
log_file = f"{folder}/kiva_model.log"
rotated_log_file = f"{folder}/kiva_model_rotated.log"

if not os.path.exists(folder):
    os.makedirs(folder)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(name)s %(levelname)s %(lineno)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(name)s %(levelname)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "simple",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": f"{log_file}",
            "mode": "a",
        },
        "rotated_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            # "level": "DEBUG",
            "filename": f"{rotated_log_file}",
            "encoding": "utf8",
            "maxBytes": 100000,
            "backupCount": 1,
        },
    },
    "loggers": {
        "": {  # root
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "overwrite": True,
}
#logging.config.dictConfig(LOGGING)

# Disable modules' loggers
# logging.getLogger("ib_insync.client").disabled = True


def getLogger(name: str):
    return logging.getLogger(name)


def overwrite_log(logging_file: str = None):
    """
    Overwrite log file. This function is required because
    LOGGING["handlers"]["file"]["mode"] = "w" (overwrite) does not work.

    Parameters
    ----------
    logging_file : str
        if None, default to LOGGING["handlers"]["file"]["filename"];
        example file: "./logs/kiva_model.log"
    """
    if LOGGING["overwrite"]:
        if logging_file is None:
            logging_file = LOGGING["handlers"]["file"]["filename"]
        with open(logging_file, "w", encoding="utf-8") as f:
            f.close()
