# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from testcases.testcase import *


class PerfOutput(TestCase):
    def __init__(self, casename, filename, tags, flag):
        self.casename = casename
        self.filename = filename
        self.tags = tags
        self.flag = flag

    # Get data from output of main_test
    def allclose(self, raw_strdata):
        return True
    
    def latency(self, raw_strdata):
        real_time = float(raw_strdata[-1].strip("\n").split(" ")[-2][:-1])
        return real_time
        #return raw_strdata[-1]


def create_perf_case(base_folder, json_data):
    testcase = json_data["testcase"]
    tags = json_data["tag"]
    filename = os.path.join(base_folder, json_data["filename"])
    flag = ""
    if "flag" in json_data:
        flag = json_data["flag"]

    return PerfOutput(testcase, filename, tags, flag)
