from typing import Any, Optional
from pipeline import PipelineComponent
from tensorflow import lite
import numpy as np


class PredictionPipeline(PipelineComponent):
    model: lite.Interpreter
    input_details: list[dict[str, Any]]
    output_details: list[dict[str, Any]]
    labels: list[str] | None

    def __init__(
        self, model_path: str, label_path: Optional[str] = None, num_threads: int = 1
    ) -> None:
        self.model = lite.Interpreter(model_path=model_path, num_threads=num_threads)

        if label_path:
            self.labels = []
            with open(label_path, "r+") as f:
                self.labels = [label.strip() for label in f.readlines()]

            print(self.labels)
        else:
            self.labels = None

        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        return

    def process(self, data: Any) -> Any:
        if not isinstance(data, np.ndarray):
            return

        if data.shape != (30, 44, 3):
            raise ValueError("Data is not of the correct shape")

        print("we are predicting, we think")

        input_data = np.array([data.reshape((30, 44, 1, 3))], dtype="float32")

        input_details_tensor_index = self.input_details[0]["index"]
        self.model.set_tensor(input_details_tensor_index, input_data)
        self.model.invoke()

        output_details_tensor_index = self.output_details[0]["index"]
        results = self.model.get_tensor(output_details_tensor_index)

        result_probability = np.max(results)
        result_ind = int(np.argmax(results))

        if self.labels:
            return self.labels[result_ind], result_probability

        return result_ind, result_probability

    def get_data(self) -> Any:
        return
