FROM fedora:26

RUN dnf -y install \
	git \
	wget \
	unzip \
	which \
	gcc

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-installer-linux-x86_64.sh
RUN sh bazel-0.5.2-installer-linux-x86_64.sh

RUN bazel --help

RUN mkdir /home/adam
RUN mkdir /home/adam/dev
RUN mkdir /home/adam/dev/tools

RUN wget https://dl.google.com/android/repository/android-ndk-r12b-linux-x86_64.zip
RUN cd /home/adam/dev/tools && unzip /android-ndk-r12b-linux-x86_64.zip



