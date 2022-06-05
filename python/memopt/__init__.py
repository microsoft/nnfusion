from .modify_output_pass import modify_output_pass
from .debug_pass import *
from .scope import get_scope, Scope
from .modify_input_pass import modify_input_pass
from .schedule_rewrite_V1 import CodeGenerator
from .utils import build_op

_verbose = 1
def set_log_level(verbose):
    global _verbose
    _verbose = verbose

def get_log_level():
    return _verbose
