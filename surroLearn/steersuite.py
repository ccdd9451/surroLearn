#!/usr/bin/env python
# encoding: utf-8

import asyncio
import zmq
import json
import numpy as np


class SteerSuite(object):
    def __init__(self, sockfilename):
        self._sock_f = "ipc://" + sockfilename

    def send(self, msg):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self._sock_f)
        socket.send_json(msg)
        return socket

    def recv(self, sock):
        return sock.recv_json()


    def update(self, params_before, params_after):
        shape = params_before.shape
        params_confirmed = np.ones(shape, dtype=np.float32)

        socks = []

        for i in range(shape[0]):
            msg = {
                "before": params_before[i, :].tolist(),
                "after":  params_after[i, :].tolist()
            }
            socks.append(self.send(msg))

        for i in range(shape[0]):
            params_confirmed[i, :] = self.recv(socks[i])

        return params_confirmed


