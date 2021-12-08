"""
Code modified from:
https://github.com/ignavier/golem/blob/main/src/utils/logger.py
"""
from datetime import datetime
import logging
import platform
import subprocess
import sys

import psutil
from pytz import timezone, utc


def setup_logger(log_path, level='INFO'):
    """Set up logger.

    Args:
        log_path (str): Path to create the log file.
        level (str): Logging level. Default: 'INFO'.
    """
    log_format = '%(asctime)s %(levelname)s - %(name)s - %(message)s'
    def custom_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone('EST')
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    logging.basicConfig(
         filename=log_path,
         level=logging.getLevelName(level),
         format= log_format,
     )

    logging.Formatter.converter = custom_time

    # Set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    # Add the console handler to the root logger
    logging.getLogger('').addHandler(console)

    # Log for unhandled exception
    logger = logging.getLogger(__name__)
    sys.excepthook = lambda *ex: logger.critical("Unhandled exception.", exc_info=ex)


def get_system_info():
    # Code modified from https://stackoverflow.com/a/58420504
    try:
        info = {}
        info['git_revision_hash'] = get_git_revision_hash()
        info['platform'] = platform.system()
        info['platform-release'] = platform.release()
        info['platform-version'] = platform.version()
        info['architecture'] = platform.machine()
        info['processor'] = platform.processor()
        info['ram'] = '{} GB'.format(round(psutil.virtual_memory().total / (1024.0 **3)))
        info['cpu_count'] = psutil.cpu_count()
        info['cpu_count'] = psutil.cpu_count()

        # Calculate percentage of available memory
        # Referred from https://stackoverflow.com/a/2468983
        info['percent_available_ram'] = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        return info
    except Exception as e:
        return None


def get_git_revision_hash():
    # Referred from https://stackoverflow.com/a/21901260
    try:
        return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())[2:-1]
    except:
        return ''