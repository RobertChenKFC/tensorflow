import argparse
import lite
import numpy as np
import tensorflow.python.framework
import representative_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert TensorFlow saved model to TensorFlow Lite model and run my pass"
    )
    parser.add_argument(
        "model_dir",
        metavar="model-dir",
        help="directory containing TensorFlow saved model",
    )
    parser.add_argument(
        "quant_model_path",
        metavar="quant-model-path",
        help="path to save converted TensorFlow Lite model",
    )
    parser.add_argument(
        "inputs_path",
        metavar="inputs-path",
        help="path to saved input numpy array",
    )
    parser.add_argument(
        "representative_dataset",
        metavar="representative-dataset",
        help="name of the representative dataset to use",
    )
    parser.add_argument(
        "--enable-flex",
        dest="enable_flex",
        action="store_true",
        help="enable select TensorFlow operations",
    )
    parser.add_argument(
        "--enable-mlir-quantizer",
        dest="enable_mlir_quantizer",
        action="store_true",
        help="enable mlir calibration and quantization",
    )
    parser.add_argument(
        "--float",
        dest="float",
        action="store_true",
        help="output float model instead of quantized model",
    )
    args = parser.parse_args()

    inputs = np.load(args.inputs_path)
    converter = lite.TFLiteConverterV2.from_saved_model(args.model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    if not args.float:
        converter.representative_dataset = (
            representative_dataset.representative_datasets[
                args.representative_dataset
            ](inputs)
        )
        if args.enable_flex:
            converter.target_spec.supported_ops = [
                lite.OpsSet.TFLITE_BUILTINS_INT8,
                lite.OpsSet.SELECT_TF_OPS,
            ]
        else:
            converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]

        if args.enable_mlir_quantizer:
            converter.experimental_new_quantizer = True
        else:
            converter.experimental_new_quantizer = False

        converter.inference_input_type = tensorflow.python.framework.dtypes.uint8
        converter.inference_output_type = tensorflow.python.framework.dtypes.uint8

    quant_model = converter.convert()
    with open(args.quant_model_path, "wb") as outfile:
        outfile.write(quant_model)
