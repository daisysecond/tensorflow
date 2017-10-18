"""
Display a vision image sample with the training data visualized.

This is how i know the training data is good.
"""

import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pylab
import numpy as np

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


def run(flags):
    if flags.image is None:
        img = Image.open("%s/%s.png" % (flags.directory, flags.sample))
    else:
        img = Image.open(flags.image)

    with open("%s/%s.csv" % (flags.directory, flags.sample), 'r') as csv:
        fs = [float(v) for v in csv.read().split(',')[:-1]]
    fv = np.array(fs).reshape((8,8, 5))
    _render(img, fv, boxtruth, dottruth, rtruth)

    if flags.prediction:
        with open(flags.prediction, 'r') as csv:
            fs = [float(v) for v in csv.read().split(',')[:-1]]
        fv = np.array(fs).reshape((8,8, 5))
        _render(img, fv, boxprediction, dotprediction, rprediction)

    plt.imshow(img)
    pylab.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/home/adam/Desktop/b',
        help='Directory load data from unity script'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=-1,
        help="""\
         Sample to show (index)
         """
    )
    parser.add_argument(
        '--image',
        help="""\
         Image to show
         """
    )
    parser.add_argument(
        '--prediction',
        help="""\
         If set then render the prediction.
         """
    )

    f, unparsed = parser.parse_known_args()
    run(f)
