from dataclasses import dataclass


@dataclass
class RealsenseCameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    output_dir: str = "/tmp"
