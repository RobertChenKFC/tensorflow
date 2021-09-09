import numpy as np
import os
import PIL.Image


class OperationRepresentativeDataset:
    name = "operation"

    def __init__(self, inputs):
        self.inputs_ = inputs

    def __call__(self):
        for cur_input in list(zip(*self.inputs_))[:100]:
            cur_input = [np.array([x]) for x in cur_input]
            yield cur_input


class RNNRepresentativeDataset:
    name = "rnn"
    batch_size = 200

    def __init__(self, inputs):
        self.inputs_ = inputs

    def __call__(self):
        for i in range(100):
            inputs = np.transpose(
                self.inputs_[i * self.batch_size: (i + 1) * self.batch_size],
                [1, 0, 2],
            )
            yield [inputs]


class AbsConvExperimentDataset:
    name = "abs_conv_experiment"

    def __init__(self, inputs):
        self.inputs_ = inputs

    def __call__(self):
        for i in range(100):
            yield [self.inputs_[i: i + 1]]


class AbsMobilenetExperimentDataset:
    name = "abs_mobilenet_experiment"

    def __init__(self, inputs):
        self.inputs_ = inputs

    def __call__(self):
        for i in range(100):
            yield [self.inputs_[i : i + 1]]


class ModelYOLOv4Dataset:
    name = "model_yolov4"
    dataset_path = "/home/robert/compiler-lab/yolov4/val2017/"

    def __init__(self, inputs):
        self.inputs_ = inputs

    def __call__(self):
        count = 0
        num_calibration_steps = 200
        while True:
            for img_filename in os.listdir(self.dataset_path):
                img_path = os.path.join(self.dataset_path, img_filename)
                img = PIL.Image.open(img_path)
                img = img.resize((416, 416))
                img = np.array(img).astype(np.float32)
                yield [np.expand_dims(img / 255, axis=0)]
                count += 1
                if count >= num_calibration_steps:
                    return


class ModelVideoCaptioningDataset:
    name = "model_video_captioning"

    def __init__(self, inputs):
        self.inputs_ = inputs

    def __call__(self):
        yield [
            np.random.random((1, 40, 300)).astype(np.float32),
            np.random.random((1, 1000)).astype(np.float32),
        ]


representative_datasets = {
    OperationRepresentativeDataset.name: OperationRepresentativeDataset,
    RNNRepresentativeDataset.name: RNNRepresentativeDataset,
    AbsConvExperimentDataset.name: AbsConvExperimentDataset,
    AbsMobilenetExperimentDataset.name: AbsMobilenetExperimentDataset,
    ModelYOLOv4Dataset.name: ModelYOLOv4Dataset,
    ModelVideoCaptioningDataset.name: ModelVideoCaptioningDataset
}
