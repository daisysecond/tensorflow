FROM fedora:26

RUN dnf -y install \
	git \
	wget \
	unzip \
	which

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-installer-linux-x86_64.sh
RUN sh bazel-0.5.2-installer-linux-x86_64.sh

RUN bazel --help
