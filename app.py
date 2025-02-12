from typing import Deque
from vision import Vision
import cv2 as cv
from llm import LLM
import numpy as np
from collections import deque
import time


a = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "Æ",
    "Ø",
    "Å",
]


class App:
    llm: LLM
    v: Vision
    time_threshold: float

    def __init__(self, time_threshold: float = 0.5):
        self.llm = LLM()
        self.v = Vision(cap_device=0)
        self.time_threshold = time_threshold

    def run(self):
        draw_debug = True
        last_hand_id: Deque[list[int]] = deque(maxlen=60)
        last_frame_time: float | None = None

        while True:
            key = cv.waitKey(10)
            # escape
            if key == 27:
                break

            # debug
            if key == 46:
                draw_debug = not draw_debug

            # space
            if key == 32:
                pred_result = self.llm.get_result()
                print("Predicted this word:", pred_result)
                last_hand_id.clear()
                last_frame_time = None
                continue

            hand_id = self.v.run_frame(draw_debug)

            if hand_id[0] != -1 or hand_id[1] != -1:
                last_hand_id.append(hand_id)
                last_frame_time = time.time()
            elif hand_id[0] == -1 and hand_id[1] == -1:
                now = time.time()

                if (
                    last_frame_time is not None
                    and now - last_frame_time > self.time_threshold
                ):
                    if len(last_hand_id) < 15:
                        last_hand_id.clear()
                    else:
                        arr = np.array(last_hand_id)
                        right = arr[:, 0]
                        left = arr[:, 1]
                        right_unique, right_count = np.unique(right, return_counts=True)
                        left_unique, left_count = np.unique(left, return_counts=True)
                        right_value = right_unique[np.argmax(right_count)]
                        left_value = left_unique[np.argmax(left_count)]

                        if right_value != -1:
                            val = a[right_value].lower()
                            print(f"Right hand value: {val}")
                            self.llm.add_letter(val)
                        if left_value != -1:
                            print(f"Left hand value: {a[left_value]}")

                        last_hand_id.clear()
                        last_frame_time = None


if __name__ == "__main__":
    app = App()
    app.run()
