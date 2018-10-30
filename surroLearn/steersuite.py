#!/usr/bin/env python
# encoding: utf-8

import asyncio
import zmq
import json
import numpy as np


class SteerSuite(object):
    def __init__(self, sockfilename):
        self._sock_f = "ipc://" + sockfilename
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self._sock_f)

        self._socket = socket

    def send(self, msg):
        self._socket.send_json(msg)
        return self._socket.recv_json()

    def update(self, params_before, params_after):
        shape = params_before.shape
        params_confirmed = np.ones(shape, dtype=np.float32)

        for i in range(shape[0]):
            msg = {
                "taskname": "cmp",
                "before": params_before[i, :].tolist(),
                "after":  params_after[i, :].tolist()
            }
            params_confirmed[i, :] = self.send(msg)

        return params_confirmed

    def init_validation(self, params_randinit):
        msg = {
            "taskname": "init",
            "params": params_randinit.ravel().tolist()
        }

        return int(self.send(msg))



