import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt


# Function to convert a time in seconds to a frame index
def time_to_frame_index(time_sec, fps):
    return int(time_sec * fps)


@dataclass
class Experiment:
    experiment_path: Path
    keyframe_times: List[float]

    def __post_init__(self) -> None:
        if not self.experiment_path.exists():
            raise RuntimeError(f"{self.experiment_path} does not exist")

        self.video_files = [
            self.experiment_path / file
            for file in ("camera_front.mp4", "camera_side.mp4", "camera_top.mp4")
        ]
        for file in self.video_files:
            if not file.exists():
                raise RuntimeError(
                    f"Could not find {file.name} in {self.experiment_path}"
                )

    def _make_frames(self, video_path: Path):
        # Read the video file
        cap = cv2.VideoCapture(str(video_path))

        # Ensure the video is opened
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract and overlay frames
        frames = []
        for idx in range(len(self.keyframe_times) - 1):
            time_sec = self.keyframe_times[idx]
            frame_index = time_to_frame_index(time_sec, fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            ret, frame1 = cap.read()
            if not ret:
                break

            time_sec_next = self.keyframe_times[idx + 1]
            frame_index = time_to_frame_index(time_sec_next, fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            ret, frame2 = cap.read()
            if not ret:
                frame2 = frame1  # If there's no next frame, use the same frame

            # Convert color to RGB
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # Overlay the second frame on the first frame with transparency
            alpha = 0.5  # Transparency factor
            overlay_frame = cv2.addWeighted(frame1, alpha, frame2, 0.8, 0)

            frames.append(overlay_frame)

        # Clean up
        cap.release()

        return frames

    def make_figures(self, save: bool = True) -> None:
        figure_folder = self.experiment_path / "figures"
        if save:
            figure_folder.mkdir(parents=True, exist_ok=True)

        for video_file in self.video_files:
            frames = self._make_frames(video_file)

            frames_per_row = len(frames)
            fig, axes = plt.subplots(
                1, frames_per_row, figsize=(5.0 * frames_per_row, 4.5)
            )  # Adjust the size as needed

            for col, frame in enumerate(frames):
                axes[col].imshow(frame)
                axes[col].axis("off")

            # Adjust subplot parameters to remove padding
            plt.subplots_adjust(wspace=0, hspace=1.2)
            plt.tight_layout()

            video_type = str(video_file.name).split(".")[0]

            if save:
                fig.savefig(str(figure_folder / f"{video_type}.pdf"))
            else:
                plt.show()


@dataclass
class ExperimentsMetadata:
    experiment_folder: str
    exps_to_use: List[str]
    keyframes: List[List[float]]

    def save(self, filepath: Path) -> None:
        metadata_as_dict = asdict(self)

        with open(filepath, "w") as f:
            json.dump(metadata_as_dict, f)

    @classmethod
    def load(cls, filepath: Path) -> "ExperimentsMetadata":
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(**data)

    @property
    def exp_dir(self) -> Path:
        return Path(self.experiment_folder) / "hardware_experiments"

    def load_experiments(self) -> List[Experiment]:
        exps = [
            Experiment(self.exp_dir / exp, keyframe)
            for exp, keyframe in zip(self.exps_to_use, self.keyframes)
        ]
        return exps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        help="Path to folder containing experiments",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    exp_folder = Path(args.dir)

    assert exp_folder.exists()

    # metadata_files = list(exp_folder.glob("**/*metadata.txt"))
    # assert len(metadata_files) == 1
    # metadata_file = metadata_files[0]

    exps_to_use = [
        "19-05-00",
        "19-08-27",
        "20-39-51",
        "20-47-40",
    ]
    keyframes = [
        [5, 9, 13.4, 23, 34, 42],
        [5, 12, 17, 36, 39, 44, 66, 71],
        [5, 9, 15, 35, 38, 39, 60, 65],
        [20, 56, 73],
    ]

    metadata = ExperimentsMetadata(str(exp_folder), exps_to_use, keyframes)
    metadata.save(exp_folder / "metadata.json")

    exps = metadata.load_experiments()
    for exp in exps:
        exp.make_figures()


if __name__ == "__main__":
    main()
