from typing import Any
from pipeline.pipeline import PipelineComponent
import numpy as np
import heapq


class DetectionWindow:
    start_timestamp: float
    end_timestamp: float
    data: np.ndarray

    def __init__(
        self, start_timestamp: float, end_timestamp: float, data: np.ndarray
    ) -> None:
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.data = data
        return

    def __eq__(self, o) -> bool:
        if not isinstance(o, DetectionWindow):
            return False

        return (
            self.start_timestamp == o.start_timestamp
            and self.end_timestamp == o.end_timestamp
        )

    def __hash__(self) -> int:
        diff = int(self.start_timestamp * 1000) + int(self.end_timestamp * 1000)

        return diff * len(self.data)


class UniqueQueue:
    "Unique queue allows a unique queue that remebers data that has been pushed to it so it cannot have historically equal data"

    queue: list
    unique_data: set[DetectionWindow]

    def __init__(self):
        self.queue = []
        self.unique_data = set()

    def push(self, data: DetectionWindow) -> None:
        "Adds data to the queue."
        if data not in self.unique_data:
            heapq.heappush(self.queue, data)
            self.unique_data.add(data)

    def pop(self) -> DetectionWindow:
        "Does not remove the unique entries to avoid duplicated data to be predicated again."
        data = heapq.heappop(self.queue)
        # self.unique_data.remove(data)
        return data

    def __len__(self) -> int:
        return len(self.queue)


class DetectionPipeline(PipelineComponent):
    vel_threshold: float
    acc_threshold: float
    min_detection_window: int
    detection_windows: UniqueQueue

    def __init__(
        self,
        vel_threshold: float = 0.1,
        acc_threshold: float = 5,
        min_detection_window: int = 5,
    ) -> None:
        self.vel_threshold = vel_threshold
        self.acc_threshold = acc_threshold
        self.min_detection_window = min_detection_window
        self.detection_windows = UniqueQueue()
        return

    def process(self, data: Any) -> Any:
        # size of this will be 1 + 49*3*3
        if not isinstance(data, np.ndarray):
            return

        if len(data) < 15:
            return

        # 148 - 295 is vel
        vel = data[:, 148:295].reshape((-1, 49, 3))

        vels = np.linalg.norm(vel, axis=2)

        vels_avg = np.mean(vels, axis=1)
        max_speeds = np.nanmax(vels, axis=1)

        custom_vel = max_speeds * vels_avg

        vels_threshold = (custom_vel >= self.vel_threshold).astype(int)

        vel_diffs = np.diff(vels_threshold)
        starts = np.where(vel_diffs == 1)[0] + 1
        ends = np.where(vel_diffs == -1)[0] + 1

        if vels_threshold[0]:
            starts = np.r_[0, starts]
        if vels_threshold[-1]:
            ends = np.r_[ends, len(vels_threshold)]

        # ensure the movement has ended before we pass it along
        for start, end in zip(starts, ends):
            length = end - start
            if length >= self.min_detection_window:
                # Check for at least 2 False values after the run (without going out of bounds)
                if (
                    end + 1 < len(vels_threshold)
                    and not vels_threshold[end]
                    and not vels_threshold[end + 1]
                ):
                    start_timestamp = data[start, 0]
                    end_timestamp = data[end, 0]

                    self.detection_windows.push(
                        DetectionWindow(
                            start_timestamp, end_timestamp, data[start:end, :]
                        )
                    )

        return

    def get_data(self) -> Any:
        if len(self.detection_windows) > 0:
            return self.detection_windows.pop()
        return
