#!/usr/bin/env python
# encoding: utf-8

import zmq
import json
import numpy as np

class SteerSuite(object):
    def __init__(self, sockfilename):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        print(sockfilename)
        self._socket.connect("ipc://"+sockfilename)

    def update(self, params_before, params_after):
        shape = params_before.shape
        params_confirmed = np.ones(shape, dtype=np.float32)
        for i in range(shape[0]):
            msg = {
                "before": params_before[i, :].tolist(),
                "after":  params_after[i, :].tolist()
            }
            self._socket.send_json(msg)
            result = self._socket.recv_json()
            params_confirmed[i, :] = result
        return params_confirmed


