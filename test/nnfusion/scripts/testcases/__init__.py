# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from testcases.testcase import *
import testcases.naive_cases
import testcases.cpu_perf_cases
import testcases.perf_cases

TestCases = list()

tests_load_funtion = {
    "naive_case_single_line": testcases.naive_cases.create_naive_case_single_line, "naive_case_multi_lines": testcases.naive_cases.create_naive_case_multi_lines, 
    "cpu_perf_case_single_line": testcases.cpu_perf_cases.create_cpu_perf_case_single_line, "cpu_perf_case_single_lines": testcases.cpu_perf_cases.create_cpu_perf_case_multi_lines,
    "perf_case" : testcases.perf_cases.create_perf_case
}

def parse_tests(base_folder, json_data):
    # read first level
    # must-have fields
    if "type" in json_data:
        type = json_data["type"]
        name = json_data["testcase"]

        if type in tests_load_funtion:
            test = tests_load_funtion[type](base_folder, json_data)
            TestCases.append(test)
            logging.info("Load testcase: " + name)

    # read list of tests
    if "testcases" in json_data:
        for test in json_data["testcases"]:
            parse_tests(base_folder, test)

# load test case from json here
def load_all_tests(models, testcase_configs):
    for root, dirs, files in os.walk(testcase_configs):
        for file in files:
            name, suf = os.path.splitext(file)
            if suf == ".json":
                file = os.path.join(root, file)
                logging.info("found " + file)
                with open(file, 'r') as f:
                    data = json.load(f)
                    parse_tests(models, data)

def load_tests(models, testcase_configs):
    global TestCases
    TestCases = list()
    load_all_tests(models, testcase_configs)
    return TestCases
