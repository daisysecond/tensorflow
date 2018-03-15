
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from time import sleep
import struct
import array
import socket


def message(v):
    v = v + b' ' * (10 - len(v))  # Pad
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 8001))
    s.send(v)
    data = None
    try:
        data = s.recv(1024)
    except ConnectionResetError:
        s.close()
    return data


def add_berry(count=1):
    message(bytes('b%s' % count, 'ascii'))


def reset():
    message(b'r')
    sleep(0.5)


def act(a=1, b=1):
    """ Returns hand_xyz, berry_xyz """
    d = message(bytes('a%s%s' % (a, b), 'ascii'))
    a = array.array('f', d[2:])
    return a


def loop():
    reset()
    add_berry(1)


    act(1, 2)


def generator(scenarios=1):
    for _ in range(scenarios):
        reset()
        add_berry(1)


if __name__ == '__main__':
    for g in generator():
        print(g)