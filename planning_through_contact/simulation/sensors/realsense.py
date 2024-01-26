from dataclasses import dataclass
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import logging

from planning_through_contact.simulation.sensors.realsense_camera_config import RealsenseCameraConfig

logger = logging.getLogger(__name__)


class RealsenseCamera:
    def __init__(self, name: str, serial_number: str, config: RealsenseCameraConfig):
        self.name = name
        self.serial_number = serial_number
        self.config = config

        self._pipeline = rs.pipeline()
        rsconfig = rs.config()
        rsconfig.enable_device(serial_number)
        rsconfig.enable_stream(
            rs.stream.color, config.width, config.height, rs.format.bgr8, config.fps
        )
        self._pipeline.start(rsconfig)

    def _get_filename(self):
        return f"{self.config.output_dir}/{self.name}.mp4"

    def _record(self, stop_event: threading.Event):
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                self._get_filename(),
                fourcc,
                self.config.fps,
                (self.config.width, self.config.height),
            )

            while not stop_event.is_set():
                frames = self._pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                out.write(color_image)

        finally:
            out.release()

    def start_recording(self):
        self.is_recording = True
        self._stop_event = threading.Event()
        self._recording_thread = threading.Thread(
            target=self._record, args=(self._stop_event,)
        )
        self._recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self._stop_event.set()
        self._recording_thread.join()
        self._pipeline.stop()
        logger.info(f"Saved recording to {self._get_filename()}")
