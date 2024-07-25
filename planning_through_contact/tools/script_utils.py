import argparse
import inspect
import logging
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Type, TypeVar

import numpy as np
import yaml

T = TypeVar("T", bound="YamlMixin")


class YamlMixin:
    """
    A mixin that adds save/load functionality to a dataclass, and makes
    sure that the output format (yaml) is also human readable, even for
    numpy arrays.
    """

    def to_dict(self) -> dict:
        if not is_dataclass(self):
            raise TypeError("save method requires a dataclass instance")

        def convert_ndarrays(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_ndarrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_ndarrays(i) for i in obj]
            else:
                return obj

        data = asdict(self)
        data = convert_ndarrays(data)
        return data

    def save(self, file_path: Path) -> None:
        import yaml

        with open(file_path, "w") as yaml_file:
            yaml.dump(
                self.to_dict(),
                yaml_file,
                default_flow_style=False,
                sort_keys=True,
            )

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        def convert_lists_to_ndarrays(obj: Any) -> Any:
            if isinstance(obj, list):
                try:
                    return np.array(obj)
                except:
                    return [convert_lists_to_ndarrays(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert_lists_to_ndarrays(v) for k, v in obj.items()}
            else:
                return obj

        data = convert_lists_to_ndarrays(data)
        config = cls(**data)
        return config

    @classmethod
    def load(cls: Type[T], file_path: Path) -> T:
        import yaml

        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        return cls.from_dict(data)


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


def get_current_git_commit() -> str:
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError:
        raise RuntimeError("Could not get current git commit")


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
