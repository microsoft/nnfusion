import os
import sys
import getpass

# config start
KERNELDB_REQUEST_FNAME="kerneldb_request.log"
TMP_DIR = f"/dev/shm/{getpass.getuser()}/controlflow"
NNFUSION_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..'))
KERNELDB_PATH = os.path.expanduser(f"/tmp/{getpass.getuser()}/kernel_cache.db")
NUM_GPU = 8
# config end

os.environ["NNFUSION_ROOT"] = NNFUSION_ROOT
os.environ["PATH"] = os.path.join(NNFUSION_ROOT, "build/src/tools/nnfusion") + ":" + os.environ["PATH"]
sys.path.insert(1, os.path.abspath(NNFUSION_ROOT + "/src/python"))

sys.path.insert(1, TMP_DIR)
os.system(f"mkdir -p {TMP_DIR}")
