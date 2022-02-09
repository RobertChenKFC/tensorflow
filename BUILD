exports_files([
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "LICENSE",
])

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and their arguments:
    targets = {
        "//tensorflow/lite/python:tflite_convert_with_mypass": "",
    },
    # For more details, feel free to look into refresh_compile_commands.bzl if you want.
)
