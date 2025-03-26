from typing import Any
from typing_extensions import override
from pipeline import PipelineComponent
import numpy as np


class LoggerPipeline(PipelineComponent):
    newest_data: Any

    def __init__(self) -> None:
        self.newest_data = None

    @override
    def process(self, data: Any) -> None:
        if isinstance(data, np.ndarray):
            print(f"Array length: {data.shape}, {data[0, 0]}")
        elif isinstance(data, dict) and isinstance(self.newest_data, dict):
            new_timestamp = data.get("timestamp", None)
            old_timestamp = self.newest_data.get("timestamp", None)
            if new_timestamp and old_timestamp:
                process_time = new_timestamp - old_timestamp
                print("Process_time:", process_time)
        elif isinstance(data, tuple):
            if len(data) == 2:
                print(f"Predicted: {data[0]} with a probability of {data[1]}")
        elif data is None:
            pass
        else:
            print(type(data), data)
        self.newest_data = data

    @override
    def get_data(self) -> Any:
        return self.newest_data
