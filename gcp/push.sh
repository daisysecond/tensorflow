#!/usr/bin/env bash
set -e

BASE_IMAGE_VERSION=v6

lines=`gcloud container images list-tags gcr.io/$PROJECT/t1 --filter="$BASE_IMAGE_VERSION" | wc -l`
if [ $lines -eq 0 ]
then
  echo "Building new base image version=$BASE_IMAGE_VERSION"
  gcloud container builds submit \
    --timeout=3h \
    --tag gcr.io/$PROJECT/t1:$BASE_IMAGE_VERSION \
    t1
fi


gcloud container builds submit \
  --timeout=2h \
  --tag gcr.io/$PROJECT/t2:$BASE_IMAGE_VERSION \
  t2
