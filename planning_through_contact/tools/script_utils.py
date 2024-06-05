import argparse
import inspect
import logging
from pathlib import Path
from typing import Optional, Tuple


def get_main_script_path() -> Optional[Path]:
    # Traverse the stack
    for frame in inspect.stack():
        # Check if the frame is the main script
        if (
            Path(frame.filename).resolve()
            == Path(inspect.stack()[-1].filename).resolve()
        ):
            return Path(frame.filename)
    return None


def make_output_folder() -> Path:
    main_script_path = get_main_script_path()
    assert main_script_path is not None
    high_level_output_dir = Path("SCRIPT_OUTPUTS")
    output_dir = high_level_output_dir / main_script_path.name.split(".")[0]

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
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
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
