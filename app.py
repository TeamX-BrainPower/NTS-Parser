from config import ProcessingConfig
from pipeline import (
    PipelineManager,
    MovementPipeline,
    InterpolationPipeline,
    CollectorPipeline,
    DetectionPipeline,
    PredictionPipeline,
    LoggerPipeline,
)

from mediapipe.tasks.python.vision import RunningMode

from processor.live_processor import LiveProsessor


def main():
    pipeline = PipelineManager()

    # pipeline layers
    collector = CollectorPipeline(max_size=300)
    movement = MovementPipeline()
    detection = DetectionPipeline()
    interpolation = InterpolationPipeline()
    predict = PredictionPipeline(
        "model/model_v1.tflite", label_path="model/labels_v1.txt"
    )
    logger = LoggerPipeline()

    pipeline.add_component(collector)
    pipeline.add_component(movement)
    pipeline.add_component(detection)
    pipeline.add_component(interpolation)
    pipeline.add_component(predict)
    pipeline.add_component(logger)

    config = ProcessingConfig(
        display_output=True,
        save_json=True,
        save_tfrecord=True,
        vision_mode=RunningMode.VIDEO,
        pipeline=pipeline,
    )

    processor = LiveProsessor(config)

    processor.process()

    return


if __name__ == "__main__":
    main()
