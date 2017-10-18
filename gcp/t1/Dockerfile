FROM fedora:26

# This is the base image.
# bazel, android-sdk, android-ndk

RUN dnf -y install \
  java-1.8.0-openjdk-headless \
  java-1.8.0-openjdk-devel \
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

