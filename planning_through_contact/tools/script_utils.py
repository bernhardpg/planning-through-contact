import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple


def make_output_folder() -> Path:
    curr_name = Path(__file__).name.split(".")[0]
    high_level_output_dir = Path("SCRIPT_OUTPUTS")
    output_dir = high_level_output_dir / curr_name

    from datetime import datetime

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H%M%S")

    unique_output_dir = output_dir / timestamp_str
    unique_output_dir.mkdir(exist_ok=True, parents=True)

    return unique_output_dir


def parse_debug_flag() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", help="Debug", action="store_true")

    args = parser.parse_args()
    debug = args.debug

    return debug


def make_default_logger(
    output_dir: Optional[Path] = None, test_logger: bool = False
) -> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the default log level (could be DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create handlers for both console and file logging
    console_handler = logging.StreamHandler()
    if output_dir is not None:
        name = str(output_dir / "script.log")
    else:
        name = "script.log"
    file_handler = logging.FileHandler(name)

    # Set the log level for each handler
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for both handlers
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log messages
    if test_logger:
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")

    return logger


def default_script_setup() -> Tuple[bool, Path, logging.Logger]:
    debug = parse_debug_flag()
    output_dir = make_output_folder()
    logger = make_default_logger(output_dir)

    return debug, output_dir, logger
