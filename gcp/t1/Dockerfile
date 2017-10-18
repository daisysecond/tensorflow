FROM fedora:26

# This is the base image.
# bazel, android-sdk, android-ndk

RUN dnf -y install \
  java-1.8.0-openjdk-headless \
  java-1.8.0-openjdk-devel \
  python2 \
  python2-devel \
  patch \
  python2-numpy \
  gcc-c++ \
	git \
	wget \
	unzip \
	which \
	gcc \
	expect

RUN wget --quiet https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-without-jdk-installer-linux-x86_64.sh \
  && sh bazel-0.5.2-without-jdk-installer-linux-x86_64.sh \
  && rm -f bazel-0.5.2-without-jdk-installer-linux-x86_64.sh

# Extracts installation and prints version
RUN bazel --help

RUN mkdir -p /home/adam/dev/tools

# ndk
RUN cd /home/adam/dev/tools \
  && wget --quiet https://dl.google.com/android/repository/android-ndk-r12b-linux-x86_64.zip \
  && unzip -q android-ndk-r12b-linux-x86_64.zip \
  && rm -f android-ndk-r12b-linux-x86_64.zip

COPY android-accept-licenses.sh /android-accept-licenses.sh
COPY packages.txt /packages.txt

RUN cd /home/adam/dev/tools \
  && wget --output-document=android-tools.zip --quiet https://dl.google.com/android/repository/tools_r25.2.3-linux.zip \
  && unzip -q android-tools.zip -d android-sdk \
  && rm -f android-tools.zip \
  && /android-accept-licenses.sh "android-sdk/tools/bin/sdkmanager --package_file=/packages.txt"

# Checkout the base branch of tensorflow and build it

RUN git clone --depth 1 -b base https://github.com/daisysecond/tensorflow.git

ENV PYTHON_BIN_PATH /usr/bin/python
ENV PYTHON_LIB_PATH /usr/lib/python2.7/site-packages
ENV TF_NEED_JEMALLOC 1
ENV TF_NEED_GCP 1
ENV TF_NEED_HDFS 0
ENV TF_ENABLE_XLA 0
ENV TF_NEED_GDR 0
ENV TF_NEED_VERBS 0
ENV TF_NEED_MPI 0

RUN cd tensorflow \
  && ./configure \
  && bazel build berry/inspect_vision
