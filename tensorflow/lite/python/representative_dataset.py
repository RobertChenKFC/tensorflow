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


class ModelRNNCellsDataset:
    name = "model_rnn_cells"
    batch_size = 100

    def __init__(self, inputs):
        self.inputs_ = inputs

    def __call__(self):
        for i in range(0, self.batch_size * 100, self.batch_size):
            yield [self.inputs_[i:i + self.batch_size]]


class ModelPersonReidentificationDataset:
    name = "model_person_identification"
    dataset_path = "/home/robert/compiler-lab/datasets/Market-1501-v15.09.15" \
                   "/query"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200
        for img_filename in os.listdir(self.dataset_path):
            if not img_filename.endswith(".png"):
                continue
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((128, 256), resample=PIL.Image.BILINEAR)
            img = np.float32(img)[np.newaxis, :, :, ::-1]
            yield [img]

            count += 1
            if count >= num_calibration_steps:
                return


class ModelPersonAttributesDataset:
    name = "model_person_attributes"
    dataset_path = "/home/robert/compiler-lab/datasets/Market-1501-v15.09.15" \
                   "/query"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200
        for img_filename in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((80, 160))
            img = np.array(img)[:, :, ::-1].astype(np.float32)
            yield [np.expand_dims(img, axis=0)]
            count += 1
            if count >= num_calibration_steps:
                return


class ModelRapidDataset:
    name = "model_rapid"

    def __init__(self, inputs):
        pass

    def __call__(self):
        for _ in range(100):
            yield [np.random.rand(1, 608, 608, 3).astype(np.float32) * 255]


class ModelTextDetectionDataset:
    name = "model_text_detection"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        for _ in range(10):
            yield [np.random.rand(1, 480, 640, 3).astype(np.float32) * 255]


class ModelU2NetDataset:
    name = "model_u2_net"
    dataset_path = "/home/robert/compiler-lab/datasets/DUTS-TE/DUTS-TE-Image"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200
        for img_filename in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((320, 320))
            img = np.float32(img)

            img = img / np.max(img)
            rgb_mean_stddev = [(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)]
            xs = np.zeros_like(img, dtype=np.float32)
            for i, (mean, stddev) in enumerate(rgb_mean_stddev):
                xs[:, :, i] = (img[:, :, i] - mean) / stddev
            yield [xs[np.newaxis, :, :, :]]

            count += 1
            if count >= num_calibration_steps:
                return


class ModelMeetSegmentation:
    name = "model_meet_segmentation"
    # Since this model doesn't have publicly available dataset, we use DUTS
    # instead
    dataset_path = "/home/robert/compiler-lab/datasets/DUTS-TE/DUTS-TE-Image"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200
        for img_filename in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((256, 144))
            img = np.array(img).astype(np.float32)
            yield [np.expand_dims(img / 255, axis=0)]
            count += 1
            if count >= num_calibration_steps:
                return


class ModelRoadSegmentation:
    name = "model_road_segmentation"
    # Since this model doesn't have publicly available dataset, we use DUTS
    # instead
    dataset_path = "/home/robert/compiler-lab/datasets/DUTS-TE/DUTS-TE-Image"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 10
        for img_filename in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((896, 512))
            img = np.array(img).astype(np.float32)
            yield [np.expand_dims(img / 255, axis=0)]
            count += 1
            if count >= num_calibration_steps:
                return


class ModelWhiteBoxCartoonization:
    name = "model_white_box_cartoonization"
    # Since this model doesn't have publicly available dataset, we use DUTS
    # instead
    dataset_path = "/home/robert/compiler-lab/datasets/DUTS-TE/DUTS-TE-Image"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 1
        for img_filename in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((720, 720))
            img = np.array(img).astype(np.float32)
            yield [np.expand_dims(img / 255, axis=0)]
            count += 1
            if count >= num_calibration_steps:
                return


class ModelOpenClosedEyeDataset:
    name = "model_open_closed_eye"
    dataset_path = "/home/robert/compiler-lab/datasets/mrlEyes_2018_01/s0001"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200
        for img_filename in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((32, 32))
            img = img.convert("RGB")
            img = np.array(img)[:, :, ::-1].astype(np.float32)
            yield [np.expand_dims(img, axis=0)]
            count += 1
            if count >= num_calibration_steps:
                return


class ModelSelfieSegmentation:
    name = "model_selfie_segmentation"
    # Since this model doesn't have publicly available dataset, we use PFCN
    # instead
    dataset_path = "/home/robert/compiler-lab/datasets/PFCN/testing/"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200
        for img_filename in os.listdir(self.dataset_path):
            if "matte" in img_filename:
                continue
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((256, 256))
            img = np.array(img).astype(np.float32)
            yield [np.expand_dims(img / 255, axis=0)]
            count += 1
            if count >= num_calibration_steps:
                return


class ModelHumanSegmentation:
    name = "model_human_segmentation"
    dataset_path = "/home/robert/compiler-lab/datasets/mini_supervisely"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200

        with open(os.path.join(self.dataset_path, "val.txt")) as infile:
            filenames = [line.split() for line in infile.read().split("\n")]

        for img_filename, _ in filenames[:-1]:
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            img = img.resize((192, 192))
            img = np.array(img).astype(np.float32)
            yield [np.expand_dims(img / 255, axis=0)]
            count += 1
            if count >= num_calibration_steps:
                return


class ModelArtisticStylePredict:
    name = "model_artistic_style_predict"
    dataset_path = "/home/robert/compiler-lab/datasets/PainterByNumbers/test"

    def __init__(self, inputs):
        pass

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200

        for img_filename in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((256, 256))
            img = np.array(img).astype(np.float32)
            yield [np.expand_dims(img / 255, axis=0)]
            count += 1
            if count >= num_calibration_steps:
                return


class ModelArtisticStyleTransfer:
    name = "model_artistic_style_transfer"
    dataset_path = "/home/robert/compiler-lab/datasets/PainterByNumbers/test"

    def __init__(self, inputs):
        self.inputs = inputs

    def __call__(self):
        # DEBUG
        count = 0
        num_calibration_steps = 200

        for cur_input, img_filename in zip(
            self.inputs,
            os.listdir(self.dataset_path)
        ):
            img_path = os.path.join(self.dataset_path, img_filename)
            img = PIL.Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((384, 384))
            img = np.array(img).astype(np.float32)
            yield [np.expand_dims(img / 255, axis=0), cur_input]
            count += 1
            if count >= num_calibration_steps:
                return


representative_datasets = {
    OperationRepresentativeDataset.name: OperationRepresentativeDataset,
    RNNRepresentativeDataset.name: RNNRepresentativeDataset,
    AbsConvExperimentDataset.name: AbsConvExperimentDataset,
    AbsMobilenetExperimentDataset.name: AbsMobilenetExperimentDataset,
    ModelYOLOv4Dataset.name: ModelYOLOv4Dataset,
    ModelVideoCaptioningDataset.name: ModelVideoCaptioningDataset,
    ModelRNNCellsDataset.name: ModelRNNCellsDataset,
    ModelPersonReidentificationDataset.name: ModelPersonReidentificationDataset,
    ModelPersonAttributesDataset.name: ModelPersonAttributesDataset,
    ModelRapidDataset.name: ModelRapidDataset,
    ModelTextDetectionDataset.name: ModelTextDetectionDataset,
    ModelU2NetDataset.name: ModelU2NetDataset,
    ModelMeetSegmentation.name: ModelMeetSegmentation,
    ModelRoadSegmentation.name: ModelRoadSegmentation,
    ModelWhiteBoxCartoonization.name: ModelWhiteBoxCartoonization,
    ModelOpenClosedEyeDataset.name: ModelOpenClosedEyeDataset,
    ModelSelfieSegmentation.name: ModelSelfieSegmentation,
    ModelHumanSegmentation.name: ModelHumanSegmentation,
    ModelArtisticStylePredict.name: ModelArtisticStylePredict,
    ModelArtisticStyleTransfer.name: ModelArtisticStyleTransfer
}
