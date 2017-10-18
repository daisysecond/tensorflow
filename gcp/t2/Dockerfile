FROM gcr.io/footprinter-2/t1:v3

RUN git clone --depth 1 -b starthere https://github.com/daisysecond/tensorflow.git
RUN cd tensorflow && bazel build berry/inspect_vision
