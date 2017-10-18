# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple image classification with Inception.

Run image classification with your model.

This script is usually used with retrain.py found in this same
directory.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.

It outputs human readable strings of the top 5 predictions along with
their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg

NOTE: To learn to use this file and retrain.py, please see:

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from PIL import Image
import numpy as np
import tensorflow as tf

import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pylab
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image', required=True, type=str, help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    type=str,
    default='/tmp/output_graph.pb',
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    required=False,
    type=str,
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    #default='Mul:0',
    default='DecodeJpeg/contents:0',

    help='Name of the input operation')
parser.add_argument(
    '--preview',
    action='store_true',
)


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()



def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Flatten tensor and save to file
    with open('prediction.csv', 'wb') as f:
        for v in predictions.flatten():
            f.write(str(v).encode('ascii'))
            f.write(b',')

    print("Wrote prediction.csv")
    # # Sort to show labels in order of confidence
    # top_k = predictions.argsort()[-num_top_predictions:][::-1]
    # for node_id in top_k:
    #   human_string = "A" #labels[node_id]
    #   score = predictions[node_id]
    #   print('%s (score = %.5f)' % (human_string, score))

    return 0


def main(argv):
  """Runs inference on an image."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  if not tf.gfile.Exists(FLAGS.image):
    tf.logging.fatal('image file does not exist %s', FLAGS.image)

  #if not tf.gfile.Exists(FLAGS.labels):
  #  tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  # load image
  image_data = load_image(FLAGS.image)
  #im = Image.open(FLAGS.image)
  #assert(im.size == (299, 299))
  #im_data = np.asarray(im, dtype=np.int32)

#  resize_shape = tf.stack([input_height, input_width])
  #resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  #resized_image = tf.image.resize_bilinear(decoded_image_4d,
  #                                         resize_shape_as_int)
  #input_mean = 128
  #input_std = 128
  #offset_image = im_data - input_mean # tf.subtract(resized_image, input_mean)
  
  #mul_image = offset_image * (1.0 / input_std) # tf.multiply(offset_image, 1.0 / input_std)
  # load labels
#  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)

  #run_graph(mul_image.reshape((1, 299, 299, 3)), FLAGS.input_layer, FLAGS.output_layer,
  run_graph(image_data, FLAGS.input_layer, FLAGS.output_layer,
            FLAGS.num_top_predictions)

  if FLAGS.preview:
      img = Image.open(FLAGS.image)

      #with open("%s/%s.csv" % (flags.directory, flags.sample), 'r') as csv:
      #    fs = [float(v) for v in csv.read().split(',')[:-1]]
      #fv = np.array(fs).reshape((8,8, 5))
      #_render(img, fv, boxtruth, dottruth, rtruth)

      with open('prediction.csv', 'r') as csv:
          fs = [float(v) for v in csv.read().split(',')[:-1]]
      fv = np.array(fs).reshape((8,8, 5))
      _render(img, fv, boxprediction, dotprediction, rprediction)

      plt.imshow(img)
      pylab.show()




# Size of a cell in pixels
C = 299/8.0

# Radius of the dot
rtruth = 5
dottruth = (100, 100, 255, 255)
boxtruth = (0, 0, 255, 255)

rprediction = 2
boxprediction = (0, 200, 255, 255)
dotprediction = (0, 150, 255, 255)

def _render(img, fv, boxc, dotc, dr, has_cutoff=0.5):
    draw = ImageDraw.Draw(img)
    for y in range(8):
        for x in range(8):
            has = fv[y,x,0] > has_cutoff
            if has:
                cx = x * C + C / 2
                cy = y * C + C / 2
                draw.ellipse((cx - dr, cy - dr, cx + dr, cy + dr), fill=dotc)

                l, r, b, t = fv[y, x, 1:]

                draw.line((x * C - l, y * C - b, (x + 1) * C - r, y * C - b), fill=boxc)
                draw.line((x * C - l, (y+1) * C - t, (x + 1) * C - r, (y+1) * C - t), fill=boxc)

                draw.line((x * C - l, y * C - b, x * C - l, (y+1) * C - t), fill=boxc)
                draw.line(((x + 1) * C - r, y * C - b, (x + 1) * C - r, (y+1) * C - t), fill=boxc)
    del draw




if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=sys.argv[:1]+unparsed)
