import numpy as np


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

    def __init__(self, inputs):
        self.inputs_ = inputs

    def __call__(self):
        for i in range(len(self.inputs_)):
            yield [self.inputs_[i: i + 1]]


representative_datasets = {
    OperationRepresentativeDataset.name: OperationRepresentativeDataset,
    RNNRepresentativeDataset.name: RNNRepresentativeDataset,
    AbsConvExperimentDataset.name: AbsConvExperimentDataset,
    AbsMobilenetExperimentDataset.name: AbsMobilenetExperimentDataset,
    ModelYOLOv4Dataset.name: ModelYOLOv4Dataset,
}
