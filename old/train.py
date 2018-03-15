from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
https://www.tensorflow.org/tutorials/layers

This script will classify each image as having a strawberry or not.
"""


def generator(train_data, train_labels):
    for d, l in zip(train_data, train_labels):
        yield {
            'img': d,
            'label': np.int32(l % 2)
        }


dir = "/home/adam/Desktop/b"


def berry_count():
    s = ''
    for f in os.listdir(dir):
        if 'csv' not in f:
            continue
        if len(f) > len(s) or (len(f) == len(s) and f > s):
            s = f
    return int(s[:-4])


def generate_berry(start, count):
    #print("generate_berry start=%s count=%s" % (start, count))
    index = 0
    while index < count:
        with open("%s/%s.csv" % (dir, index + start), 'r') as csv:
            has_berry, x, y, z = [int(v) for v in csv.readline().split(',')[:-1]]

        im = Image.open("%s/%s.png" % (dir, index + start))

        # mnist has the image as a flat array of float32 shape (28*28)
        img = np.frombuffer(im.tobytes(), dtype=np.uint8).astype(np.float32)
        # only use red component for now
        redimg = img[::3]
        #if index < 11:
        #    print("GEN", index+start, has_berry, x/16,y/16,z/16)
        yield {
            'index': np.int32(index+start),
            'img': redimg,
            'label': np.array([has_berry and 1 or 0, x / 16, y / 16, z / 16], dtype=np.uint8)
        }
        index += 1


def main(*argv):

    # Load training and eval data
    #mnist = learn.datasets.load_dataset("mnist")
    #train_data = mnist.train.images # Returns np.array
    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    #eval_data = mnist.test.images # Returns np.array
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    count = berry_count()
    eval_count = 100

    print("Train: %s Eval %s" % (count - eval_count, eval_count))

    # Create the Estimator
    mnist_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    tensors_to_log = {
        #    "h": "HEY"
    } #{"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50,
        # formatter=lambda f: np.array_str(f['h'][0:10, :], max_line_width=2000)
    )

    if len(argv[0]) > 1:
        #sess = tf.Session()

        for i in range(1250, 1260):
            for p in mnist_classifier.predict(input_fn=generator_input_fn(lambda: generate_berry(i, 1), batch_size=1, num_epochs=1)):
                #xpos = tf.argmax(prediction['probabilities']).eval(session=sess)
                #import pdb
                #pdb.set_trace()
                for real in generate_berry(i, 1):
                    pos = p['y/n'] and ','.join([str(s) for s in [p['x'], p['y'], p['z']]]) or ''
                    print("PREDICTION", i, "y/n=", p['y/n'], "p=", pos, real['label'])

                break
        return

    # Train the model
    mnist_classifier.fit(
        input_fn=generator_input_fn(lambda: generate_berry(0, count - eval_count), target_key='label',
                                    batch_size=100, num_epochs=None, shuffle=True),
        steps=20000,
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="x"),
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        input_fn=generator_input_fn(
            lambda: generate_berry(count - eval_count, eval_count),
            target_key='label'
        ),
        metrics=metrics
    )
    print(eval_results)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features['img'], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=16 * 3 + 2)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        #session = tf.InteractiveSession()
        #tf.train.start_queue_runners(session)

        yn_label = tf.one_hot(indices=labels[:, 0], depth=2)
        x_label = tf.one_hot(indices=labels[:, 1], depth=16)
        y_label = tf.one_hot(indices=labels[:, 2], depth=16)
        z_label = tf.one_hot(indices=labels[:, 3], depth=16)
        cc = tf.concat([yn_label, x_label, y_label, z_label], axis=1)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=cc, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

    # Generate Predictions
    predictions = {
        "x": tf.argmax(input=logits[:,2:18], axis=1),
        "y": tf.argmax(input=logits[:,18:34], axis=1),
        "z": tf.argmax(input=logits[:,34:50], axis=1),
        "y/n": tf.greater(logits[:, 1], tf.fill([100, 1], 0.6)),
        "raw": tf.to_float(logits)
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


if __name__ == "__main__":
    tf.app.run()

