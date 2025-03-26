from typing import Any
from scipy.interpolate import interp1d
from pipeline import PipelineComponent
import numpy as np

from pipeline.detetion_pipeline import DetectionWindow


class InterpolationPipeline(PipelineComponent):
    interpolation_size: int

    def __init__(self, interpolation_size: int = 30) -> None:
        self.interpolation_size = interpolation_size
        self.__data_columns = [2, 3, 4, 5] + \
            list(range(8, 28)) + list(range(29, 49))
        return

    def process(self, data: Any) -> Any:
        if not isinstance(data, DetectionWindow):
            return None

        data_len = len(data.data)

        if data_len < 5:
            return None

        print("Interpolating")

        original_indices = np.linspace(0, 1, data_len)

        new_indicies = np.linspace(0, 1, self.interpolation_size)

        interpolated_frames = np.zeros((self.interpolation_size, 49 * 3))

        for point in range(1, 49 * 3 + 1):
            coord_values = data.data[:, point]
            interp_func = interp1d(
                original_indices,
                coord_values,
                kind="cubic",
                axis=0,
                fill_value="extrapolate",  # pyright: ignore
            )
            interpolated_frames[:, point - 1] = interp_func(new_indicies)

        return_data = interpolated_frames.reshape((-1, 49, 3))
        return_data = return_data[:, self.__data_columns]

        return return_data

    def get_data(self) -> Any:
        return None
