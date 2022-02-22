import argparse
import lite
import tensorflow.python.framework
from receiver_representative_dataset import ReceiverRepresentativeDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert TensorFlow saved model to TensorFlow Lite model "
                    "and run my pass"
    )
    parser.add_argument(
        "model_dir",
        metavar="model-dir",
        help="directory containing TensorFlow saved model",
    )
    parser.add_argument(
        "tflite_model_path",
        metavar="tflite-model-path",
        help="path to save converted TensorFlow Lite model",
    )
    parser.add_argument(
        "--representative-dataset-address",
        dest="representative_dataset_address",
        help="address to the sender representative dataset. If given, "
             "the model will be quantized, otherwise the model remains float.",
    )
    parser.add_argument(
        "--enable-flex",
        dest="enable_flex",
        action="store_true",
        help="enable select TensorFlow operations",
    )
    parser.add_argument(
        "--enable-old-quantizer",
        dest="enable_old_quantizer",
        action="store_true",
        help="enable old quantizer instead of MLIR quantizer",
    )
    parser.add_argument(
        "--enable-uint8",
        dest="enable_uint8",
        action="store_true",
        help="use input and output data type uint8 instead of int8. This "
             "option has no effect if --representative-dataset-address is not"
             "also specified."
    )
    args = parser.parse_args()

    converter = lite.TFLiteConverterV2.from_saved_model(args.model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    if args.representative_dataset_address:
        converter.representative_dataset = ReceiverRepresentativeDataset(
            args.representative_dataset_address
        )
        if args.enable_flex:
            converter.target_spec.supported_ops = [
                lite.OpsSet.TFLITE_BUILTINS_INT8,
                lite.OpsSet.SELECT_TF_OPS,
            ]
        else:
            converter.target_spec.supported_ops = [
                lite.OpsSet.TFLITE_BUILTINS_INT8
            ]

        converter.experimental_new_quantizer = not args.enable_old_quantizer
        if args.enable_uint8:
            dtype = tensorflow.python.framework.dtypes.uint8
        else:
            dtype = tensorflow.python.framework.dtypes.int8
        converter.inference_input_type = dtype
        converter.inference_output_type = dtype

    tflite_model = converter.convert()
    with open(args.tflite_model_path, "wb") as outfile:
        outfile.write(tflite_model)
