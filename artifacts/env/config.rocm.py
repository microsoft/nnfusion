import os
import sys
import getpass

# config start
KERNELDB_REQUEST_FNAME="kerneldb_request.log"
NNFUSION_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..'))
TMP_DIR = os.path.join(NNFUSION_ROOT, "artifacts/models/tmp")
KERNELDB_PATH = os.path.expanduser("~/.cache/nnfusion/kernel_cache.db")
NUM_GPU = 1
# config end

os.environ["NNFUSION_ROOT"] = NNFUSION_ROOT
os.environ["PATH"] = os.path.join(NNFUSION_ROOT, "build/src/tools/nnfusion") + ":" + os.environ["PATH"]
sys.path.insert(1, os.path.abspath(NNFUSION_ROOT + "/src/python"))

sys.path.insert(1, TMP_DIR)
os.system(f"mkdir -p {TMP_DIR}")
