FROM fedora:26

RUN dnf -y install \
	git \
	wget

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-installer-linux-x86_64.sh
RUN sh bazel-0.5.2-installer-linux-x86_64.sh

#RUN dnf -y copr enable vbatts/bazel
#RUN dnf -y install bazel

#RUN bazel --help

# Clone repo & build

# RUN git clone --depth 1 -b starthere https://github.com/daisysecond/tensorflow.git

# RUN cd tensorflow && bazel build berry/inspect_vision


