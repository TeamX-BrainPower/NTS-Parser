from typing import Any, Deque
from pipeline.pipeline import PipelineComponent
from collections import deque
import numpy as np


class DetectionPipeline(PipelineComponent):
    vel_threshold: float
    acc_threshold: float
    detection_windows: Deque[np.ndarray]

    def __init__(
        self,
        vel_threshold: float = 0.05,
        acc_threshold: float = 0.05,
        maxsize: int = 10,
    ) -> None:
        self.vel_threshold = vel_threshold
        self.acc_threshold = acc_threshold
        self.detection_windows = deque(maxlen=maxsize)
        return

    def process(self, data: Any) -> Any:
        # size of this will be 1 + 49*3*3
        # 148 - 295 is vel
        # 295 - 442 is acc
        if isinstance(data, np.ndarray):
            print(len(data))
            if len(data) < 15:
                return

            vel = np.nan_to_num(data[:, 148:295].reshape((-1, 49, 3)))

            acc = np.nan_to_num(data[:, 295:442].reshape((-1, 49, 3)))

            # vels_avg = np.mean(vel, axis=2)

            vels = np.linalg.norm(vel, axis=2)
            accs = np.linalg.norm(acc, axis=2)

            vels_avg = np.mean(vels, axis=1)
            max_speeds = np.nanmax(vels, axis=1)

            max_accs = np.nanmax(accs, axis=1)
            accs_avg = np.mean(accs, axis=1)

            # max_speeds_avg = np.nanmax(vels_avg, axis=1)
            # print(max_speeds.shape)
            # max_accs = np.nanmax(accs, axis=1)
            print("max speeds")
            print(max_speeds)
            print("avg speeds")
            print(vels_avg)
            print("mult")
            print(max_speeds * vels_avg)

            print("max accs")
            print(max_accs)

            print("avg accs")
            print(accs_avg)

            print("accs mult")
            print(max_accs * accs_avg)

            pass

        return

    def get_data(self) -> Any:
        if len(self.detection_windows) > 0:
            return self.detection_windows.popleft()
        return
