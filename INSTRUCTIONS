# Bazel

Use bazel 0.5.2
(0.5.3 is broken)

Download installer from github releases

https://github.com/bazelbuild/bazel/releases/tag/0.5.2

# Ensure this repo was cloned using recursive

git clone --recursive https://github.com/tensorflow/tensorflow

# 1. Generate png / csv pairs from unity.

You can inspect the generated data using:

bazel run //berry/inspect_vision

# Vision training.
# This will generate bottleneck files and then run tensorflow training.
# Files are written in /home/adam/dev/berry/data

bazel run tensorflow/examples/image_retraining:retrain -- \
 --how_many_training_steps 10000

# Ensure android sdk is installed as per ./WORKSPACE

# Build android example

