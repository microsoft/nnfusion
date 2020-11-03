# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Author: wenxh
# This script is to be used as batch system intergration test in Azure Build Agent
import os
import sys
import subprocess
import multiprocessing
import logging
import json
import pprint
import numpy as np
import testcases
import evaluator
import socket


class TestsManager:
    def __init__(self):
        config_json = "config.json"
        if len(sys.argv) == 2:
            config_json = sys.argv[1]
        self.load_config(config_json)

        # overwrite if config.json specified
        if len(sys.argv) > 2:
            self.models = sys.argv[1]
            self.nnfusion_cli = sys.argv[2]

        if not os.path.exists(self.models):
            self.models = self.load_default_models_path()

        if not os.path.exists(self.nnfusion_cli):
            self.nnfusion_cli = self.load_default_nnfusion_cli()

        if not os.path.exists(self.testcase_configs):
            self.testcase_configs= os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "./testcase_configs")

        self.capability = set()
        self.capability_detect()

        # is a dict()
        self.enabled_tags = self.user_enabled_tags

        logging.info("models folder = " + self.models)
        logging.info("testcase configs folder = " + self.testcase_configs)
        logging.info("nnfusion cli = " + self.nnfusion_cli)
        logging.info("device capability = " + ",".join(list(self.capability)))
        logging.info("enabled tags = " + str(self.enabled_tags))

    def load_config(self, config_json):
        self.user_device_capability = set()
        self.user_enabled_tags = dict()
        self.models = ""
        self.nnfusion_cli = ""
        self.nnfusion_args = ""
        self.testcase_configs = ""

        if not os.path.exists(config_json):
            config_json = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), config_json)
            if not os.path.exists(config_json):
                return

        logging.info("load config from: " + config_json)
        with open(config_json, 'r') as f:
            data = json.load(f)

            # env operations
            if "env" in data.keys():
                env_ops = data["env"]
                for env in env_ops.keys():
                    # set, append, clear, etc ...
                    if 'set' in env_ops[env].keys():
                        os.environ[env] = str(env_ops[env]['set'])
                    if 'append' in env_ops[env].keys():
                        if os.getenv(env) is None:
                            os.environ[env] = str(env_ops[env]['append'])
                        else:
                            os.environ[env] = os.getenv(
                                env) + ":" + str(env_ops[env]['append'])
                    if 'clear' in env_ops[env].keys():
                        os.environ[env] = ""
                    if 'del' in env_ops[env].keys():
                        if env in os.environ.keys():
                            del os.environ[env]

                    logging.info("\t" + env + " = " + str(os.environ[env]))

            if "device_capability" in data.keys():
                self.user_device_capability = set(data["device_capability"])

            if "enabled_tags" in data.keys():
                self.user_enabled_tags = data["enabled_tags"]

            if "models" in data.keys():
                self.models = data["models"]

            if "nnfusion_cli" in data.keys():
                self.nnfusion_cli = data["nnfusion_cli"]

            if "nnfusion_args" in data.keys():
                self.nnfusion_args = data["nnfusion_args"]

            if "testcase_configs" in data.keys():
                self.testcase_configs = data["testcase_configs"]
                

    def load_default_nnfusion_cli(self):
        nnf_clis = [os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "../../../build/src/tools/nnfusion/nnfusion"), "/usr/local/bin/nnfusion"]
        for nnf in nnf_clis:
            if os.path.exists(nnf):
                print("NNFusion CLI detected: " + nnf)
                return nnf
        logging.error("No nnfusion cli available.")
        exit(1)

    def load_default_models_path(self):
        models_path = [os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../../../../frozenmodels"), os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../../../models/frozenmodels")]
        for models in models_path:
            if os.path.exists(models):
                print("models/ folder detected: " + models)
                return models
        logging.error("No models folder available.")
        exit(1)

    def capability_detect(self):
        # Detect Cuda
        if os.path.exists("/usr/local/cuda/bin/nvcc"):
            self.capability.add("CUDA")
            logging.info("NVCC is existed.")

        if os.path.exists("/opt/rocm/bin/hcc"):
            self.capability.add("ROCM")
            logging.info("HCC is existed.")

        self.capability.add("CPU")

        if len(self.user_device_capability) > 0:
            self.capability = self.capability.intersection(
                self.user_device_capability)

    def load_test_cases(self, enabled_tags=set("correctness")):
        tests = testcases.load_tests(self.models, self.testcase_configs)
        newlist = []
        for test in tests:
            avail = False
            for tag in test.tags:
                if tag in enabled_tags:
                    avail = True
                    break
            if avail:
                newlist.append(test)
        return newlist

    def report(self):
        manager = multiprocessing.Manager()
        report_list = manager.list()
        jobs = []
        for dev in self.capability:
            p = multiprocessing.Process(target=evaluator.E2EExecutor, args=(
                self.load_test_cases(self.enabled_tags[dev]), dev, report_list, self.nnfusion_cli, self.nnfusion_args))
            jobs.append(p)
            p.start()

        if 'SIDECLI' in os.environ:
            p = multiprocessing.Process(
                target=evaluator.CLIExecutor, args=("", report_list))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        hostname = socket.gethostname()
        print("=========================================\n\n")
        print(hostname + "\tE2E Test report")
        print("\n\n=========================================\n")
        report = ("\n".join(report_list))
        print(report)
        if "Failed" in report:
            return -1
        return 0


if __name__ == "__main__":
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=LOGLEVEL)

    _m = TestsManager()
    exit(_m.report())
