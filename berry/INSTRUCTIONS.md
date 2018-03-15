# Bazel

Use bazel 0.11.1 copr

```bash
sudo dnf copr enable vbatts/bazel
sudo dnf install -y bazel
```

# Ensure this repo was cloned using recursive

git clone --recursive https://github.com/tensorflow/tensorflow

# 

You can inspect the generated data using:

```
bazel run //berry/inspect_vision
```

# Vision training.
# This will generate bottleneck files and then run tensorflow training.
# Files are written in /home/adam/dev/berry/data

```
bazel run tensorflow/examples/image_retraining:retrain -- \
 --how_many_training_steps 10000
```

# Android

 1. Ensure android sdk is installed as per ./WORKSPACE
 2. Build android example

```
bazel build -c opt //tensorflow/examples/android:tensorflow_demo
```

# Unity

https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/

1. Generate png / csv pairs from unity.