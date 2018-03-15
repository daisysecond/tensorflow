# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from Config import Config

from gym import spaces
import gym
import numpy as np
import socket
import array
from numpy.linalg import norm
import subprocess
import os
from time import sleep


class ArmGame(gym.Env):
    port = 8002

    def __init__(self, display):
        self.port = ArmGame.port
        self.display = display
        self.created = False
        ArmGame.port += 1

        # Each of the 3 motors can be (ccw, off, cw)
        self.action_space = spaces.Discrete(3**3)
        self.observation_space = spaces.Box(low=8, high=8, shape=(12,))

        self.reward = [False, False, False]
        self.step_count = 0

    def _create_and_wait_for_port(self):
        env = os.environ.copy()
        env.update({'BERRY_PORT': str(self.port)})

        if True:
            if self.display:# or self.port == 8012:
                exe = "/home/adam/Desktop/berryBuild/build1.x86"
            else:
                exe = "/home/adam/Desktop/berryHeadless/build1.x86"
            self.proc = subprocess.Popen(exe, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            self.port = 8001

        for _ in range(10):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setblocking(1)
                s.connect(('localhost', self.port))
                return
            except ConnectionRefusedError:
                sleep(2)

    def reset(self):
        if not self.created:
            self.created = True
            self._create_and_wait_for_port()

        self.reward = [False, False, False]
        self.step_count = 0

        # Reset scene and add berries
        self._send(b'r', wait=True)
        self._send(bytes('b%s' % 1, 'ascii'))
        # Initial null action to get observation
        data = self._send(bytes('a111', 'ascii'))
        collected, ar, cam = self._make_observation(data)
        self.reward = [False, False, False]
        return cam

    def _make_observation(self, d):
        collected = d[2]
        a = array.array('f', d[4:])
        # a = (hand_xyz, hand_pqd, berry_xyz, berry_pqd)
        ar = np.array(a)

        # The camera only sees hand_pq and berry_pq.
        cam = np.concatenate((ar[3:5], ar[9:11])) - 0.5
        return collected, ar, cam

    def step(self, action):

        # Convert action (scalar) into motor actions
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.unravel_index.html
        a, b, c = [np.asscalar(v) for v in  np.unravel_index([action], (3,3,3))]

        #print("ACT AS", a,b,c, 'a%s%s%s' % (a, b, c))
        data = self._send(bytes('a%s%s%s' % (a, b, c), 'ascii'))

        collected, observation, cam = self._make_observation(data)

        # Do at the start so we have one step after the collection. (work around ga3c bug)
        done = self.reward[2] == True

        reward = 0
        hand_xyz = observation[0:3]
        berry_xyz = observation[6:9]

        distance = norm(hand_xyz - berry_xyz)

        if not self.reward[0] and distance < 2:
            reward += 1
            self.reward[0] = True

        if not self.reward[1] and distance < 1:
            reward += 1
            self.reward[1] = True

        if not self.reward[2] and collected:
            reward += 1
            self.reward[2] = True

        self.step_count +=1

        done = done or self.step_count > 100
        info = None

        #print(self.step_count, hand_xyz, berry_xyz, distance, reward, done)
        return cam, reward, done, info

    def _send(self, v, wait=True):

        v = v + b' ' * (10 - len(v))  # Pad

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(1)
        s.connect(('localhost', self.port))
        data = None
        try:
            s.send(v)
            if wait:
                data = s.recv(1024)
        finally:
            s.close()
        return data


class GameManager:
    def __init__(self, game_name, display):
        self.game_name = game_name
        self.display = display

        self.env = ArmGame(display)

    def reset(self):
        observation = self.env.reset()
        return observation

    def step(self, action):
        #self._update_display()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    # def _update_display(self):
    #     if self.display:
    #         self.env.render()
