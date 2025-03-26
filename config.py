from dataclasses import dataclass
from typing import Callable, Optional
from mediapipe.tasks.python.vision import RunningMode

from pipeline import PipelineManager


@dataclass
class ProcessingConfig:
    face_model_path: str = "models/face_landmarker.task"
    hand_model_path: str = "models/hand_landmarker.task"
    pose_model_path: str = "models/pose_landmarker.task"
    output_dir: str = "data/raw"
    num_faces: int = 1
    num_hands: int = 2
    display_output: bool = True
    save_json: bool = False
    save_tfrecord: bool = False
    cap_device: int = 0
    cap_width: int = 960
    cap_height: int = 540
    vision_mode: RunningMode = RunningMode.VIDEO  # pyright: ignore
    face_callback: Optional[Callable] = None
    hand_callback: Optional[Callable] = None
    pose_callback: Optional[Callable] = None
    pipeline: Optional[PipelineManager] = None
    recording: bool = False
    fps_cap: int = 30
