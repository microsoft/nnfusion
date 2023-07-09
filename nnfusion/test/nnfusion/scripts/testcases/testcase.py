# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import logging
import os
import sys
import json


class TestCase:
    def __init__(self, casename, ground_truth, filename, tags, flag):
        self.casename = casename
        self.ground_truth = ground_truth
        self.rtol = 1.e-2
        self.atol = 1.e-2
        self.filename = filename
        self.tags = tags
        self.flag = flag

    def get_filename(self):
        return self.filename

    def allclose(self, result):
        if np.allclose(result, self.ground_truth, rtol=self.rtol, atol=self.atol):
            return True
        logging.error("%s has wrong result." % (self.casename))
        return False

    def valid(self):
        if not os.path.exists(self.get_filename()):
            logging.error("%s file not existed." % (self.get_filename()))
            return False
        return True
