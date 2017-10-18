# Description:
# Transfer learning example for TensorFlow.

load("//tensorflow:tensorflow.bzl", "py_test")

py_binary(
    name = "t",
    srcs = [
        "t.py",
    ],
    srcs_version = "PY2AND3",
    #visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "inspect_vision",
    srcs = [
        "inspect_vision.py",
    ],
    srcs_version = "PY2AND3",
    #visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow:tensorflow_py",
        #      "//tensorflow/python:framework",
        #      "//tensorflow/python:framework_for_generated_wrappers",
        #      "//tensorflow/python:graph_util",
        #      "//tensorflow/python:platform",
        #    "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)
