import cv2 as cv
import mediapipe as mp
import copy
import itertools
from draw_utils import (
    calc_bounding_rect,
    draw_bounding_rect,
    draw_landmarks,
    draw_info_text,
)

from model import KeyPointClassifier


class Vision:
    cap: cv.VideoCapture
    hands: mp.solutions.hands.Hands  # pyright: ignore
    keypoint_classifier: KeyPointClassifier
    signs: list[str]

    def __init__(
        self,
        cap_device: int = 0,
        cap_width: int = 960,
        cap_height: int = 540,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        num_threads: int = 1,
    ) -> None:
        self.cap = cv.VideoCapture(cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        mp_hands = mp.solutions.hands  # pyright: ignore
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier(
            "model/keypoint_classifier.tflite", num_threads=num_threads
        )

        self.signs = ["-1"] * 39

        with open("model/keypoint_classifier_label.csv", "r+") as f:
            self.signs = [a.strip() for a in f.readlines()]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    #
    def run_frame(self, draw_debug: bool = False) -> list[int]:
        hand_sign_id = [-1, -1]

        ret, image = self.cap.read()

        # if we get no image
        if not ret:
            raise Exception("No image found")

        image = cv.flip(image, 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Get results
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                landmark_list = self.calc_landmark_list(image, hand_landmarks)
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)

                hand_id = self.keypoint_classifier(pre_processed_landmark_list)

                hand = handedness.classification[0].label[0:].lower()

                if hand == "right":
                    hand_sign_id[0] = hand_id
                elif hand == "left":
                    hand_sign_id[1] = hand_id

                if draw_debug:
                    brect = calc_bounding_rect(image, hand_landmarks)

                    image = draw_bounding_rect(True, image, brect)
                    image = draw_landmarks(image, landmark_list)
                    image = draw_info_text(
                        image, brect, handedness, self.signs[hand_id]
                    )

        cv.imshow("Test", image)

        return hand_sign_id

    pass
