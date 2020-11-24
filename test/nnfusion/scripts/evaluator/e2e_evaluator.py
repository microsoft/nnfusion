# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import uuid, os, logging, sys, multiprocessing, tempfile

class E2EEvaluator:
    def __init__(self, testcase, codegen_folder = "cuda_codegen", default_device = "CUDA", working_foler = ".", nnfusion_cli = "", nnfusion_cli_arg = ""):
        self.codegen_folder = codegen_folder
        self.default_device = default_device
        self.testcase = testcase
        self.working_foler = working_foler
        if not os.path.exists(nnfusion_cli):
            self.nnfusion_cli = self.load_default_nnfusion_cli()
        else:
            self.nnfusion_cli = nnfusion_cli
        self.nnfusion_cli_arg = nnfusion_cli_arg
    
    def load_default_nnfusion_cli(self):
        nnf_clis = [os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "../../../build/src/tools/nnfusion/nnfusion"), "/usr/local/bin/nnfusion"]
        for nnf in nnf_clis:
            if os.path.exists(nnf):
                print("NNFusion CLI detected: " + nnf)
                return nnf
        logging.error("No nnfusion cli available.")
        exit(1)

    def nnfusion_compile(self):
        logging.info("Compiling " + self.testcase.get_filename())

        name, suf = os.path.splitext(self.testcase.get_filename())
        modeltype = "-f tensorflow"
        if suf == ".onnx":
            modeltype = "-f onnx"
        else:
            if suf == ".pt":
                modeltype = "-f torchscript"

        logging.info("cd %s && %s %s %s %s -fdefault_device=%s %s >> nnfusion.log" %
                (self.working_foler, self.nnfusion_cli, self.testcase.get_filename(), modeltype, self.testcase.flag, self.default_device, self.nnfusion_cli_arg))
        
        os.system("cd %s && %s %s %s %s -fdefault_device=%s %s >> nnfusion.log" %
                (self.working_foler, self.nnfusion_cli, self.testcase.get_filename(), modeltype, self.testcase.flag, self.default_device, self.nnfusion_cli_arg))
        if not os.path.exists("%s/nnfusion_rt/%s/nnfusion_rt.h"%(self.working_foler, self.codegen_folder)):
            logging.error("Failed at nnfusion compiling phase.")
            return False
        return True
    
    def build(self):
        os.system("cd %s/nnfusion_rt/%s/ && cmake . >> cmake.log && make -j 2>&1 >> cmake.log"%(self.working_foler, self.codegen_folder))
        if not os.path.exists("%s/nnfusion_rt/%s/main_test"%(self.working_foler, self.codegen_folder)):
            logging.error("Failed at compiling phase.")
            return False
        return True
    
    def allclose(self):
        code = os.system("cd %s/nnfusion_rt/%s/ && ./main_test > result.txt"%(self.working_foler, self.codegen_folder))
        if code != 0:
            logging.error("%s execution failed."%self.testcase.casename)
            return False
        if not os.path.exists("%s/nnfusion_rt/%s/result.txt"%(self.working_foler, self.codegen_folder)):
            logging.error("Failed at compiling phase.")
            return False
        result_file = open("%s/nnfusion_rt/%s/result.txt"%(self.working_foler, self.codegen_folder))
        results = result_file.readlines()
        if not self.testcase.allclose(results):
            logging.error("%s result missmatch."%self.testcase.casename)
            return False
        return True 

    def report(self):
        os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
        if not self.nnfusion_compile():
            os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
            return False
        if not self.build():
            os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
            return False
        if not self.allclose():
            os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
            return False
        os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
        return True

configs = {"CUDA" : ["cuda_codegen", "CUDA"], "ROCM" : ["rocm_codegen", "ROCm"], "CPU" : ["cpu_codegen","CPU"]}

def E2EExecutor(TestCases, devname, report_list, nnf, nnf_args):
    tmpdir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    logging.info("create " + tmpdir)
    os.mkdir(tmpdir) # working folder

    for test in TestCases:
        logging.info("Testing " + test.casename)
        if test.valid():
            eval = E2EEvaluator(test, configs[devname][0], configs[devname][1], tmpdir, nnf, nnf_args)
            report = devname + "\t" + test.casename + '\t' + ",".join(test.tags) + "\t";
            if eval.report():
                report += "Succeed!"
            else:
                eval = E2EEvaluator(test, configs[devname][0], configs[devname][1], tmpdir)
                if eval.report():
                    report += "Succeed!"
                else:
                    report += "Failed"
            logging.info(report)
            report_list.append(report)
    # clean
    logging.info("remove " + tmpdir)
    os.system("rm -rf %s"%tmpdir)

def CLIExecutor(info, report_list):
    print(info)
    side_cli = str(os.environ.get('SIDECLI', ''))
    if os.system(side_cli) == 0:
       report_list.append(side_cli + "\tSucceed!") 
    else:
       report_list.append(side_cli + "\tFailed") 