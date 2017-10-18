FROM gcr.io/footprinter-2/t1:v5

# Checkout the specific branch and build.
#
RUN cd tensorflow \
  && git checkout starthere \
  && bazel build berry/inspect_vision
