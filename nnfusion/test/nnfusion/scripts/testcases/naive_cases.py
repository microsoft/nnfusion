# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from testcases.testcase import *


class TestSingleOutput(TestCase):
    def __init__(self, casename, strdata, filename, tags, flag):
        self.casename = casename
        self.ground_truth = [float(v.strip()) for v in strdata.split("..")[
            0].strip().split(" ")]
        self.rtol = 1.e-4
        self.atol = 1.e-4
        self.filename = filename
        self.tags = tags
        self.flag = flag

    # Get data from output of main_test
    def allclose(self, raw_strdata):
        floatdata = [float(v.strip())
                     for v in raw_strdata[1].split("..")[0].strip().split(" ")]
        return TestCase.allclose(self, floatdata)


class TestMultiOutput(TestCase):
    def __init__(self, casename, strdata, filename, tags, flag):
        self.casename = casename
        self.ground_truth = self.extract_data(strdata.split("\n"))
        self.rtol = 1.e-2
        self.atol = 1.e-2
        self.filename = filename
        self.tags = tags
        self.flag = flag

    def extract_data(self, strs):
        data = list()
        for i in range(1, len(strs), 2):
            try:
                it = [float(v.strip())
                        for v in strs[i].strip().split("..")[0].strip().split(" ")]
                data.append(it)
            except:
                print("Unsupported output: " + strs[i].strip())
                break
        return data

    def all_allclose(self, a, b):
        cnt = 0
        for u in a:
            flag = False
            for v in b:
                if len(u) == len(v) and np.allclose(u, v, rtol=self.rtol, atol=self.atol):
                    flag = True
            if not flag:
                print("Mismatch#%d: %s" % (cnt, u))
                return False
            cnt += 1
        return True

    def allclose(self, raw_strdata):
        if not self.all_allclose(self.extract_data(raw_strdata), self.ground_truth):
            logging.error("%s has wrong result." % (self.casename))
            return False
        return True


def create_naive_case_single_line(base_folder, json_data):
    testcase = json_data["testcase"]
    output = json_data["output"]
    tags = json_data["tag"]
    filename = os.path.join(base_folder, json_data["filename"])
    flag = ""
    if "flag" in json_data:
        flag = json_data["flag"]
    return TestSingleOutput(testcase, output, filename, tags, flag)


def create_naive_case_multi_lines(base_folder, json_data):
    testcase = json_data["testcase"]
    output = "\n".join(json_data["output"])
    tags = json_data["tag"]
    filename = os.path.join(base_folder, json_data["filename"])
    flag = ""
    if "flag" in json_data:
        flag = json_data["flag"]
    return TestMultiOutput(testcase, output, filename, tags, flag)

