FROM gcr.io/footprinter-2/t1:v5

RUN git clone --depth 1 -b starthere https://github.com/daisysecond/tensorflow.git

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

