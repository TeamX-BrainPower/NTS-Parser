from typing import Any
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    model_path: str
    num_threads: int
    interpreter: tf.lite.Interpreter
    input_details: list[dict[str, Any]]
    output_details: list[dict[str, Any]]

    def __init__(self, model_path: str, num_threads: int = 1) -> None:
        self.model_path = model_path
        self.num_threads = num_threads

        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_path, num_threads=self.num_threads
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_details_tensor_index, np.array(
                [landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]["index"]

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
