import torch
from .IRpass import *
from .scope import get_scope, Scope
from .scheduler import Scheduler
from . import utils
from .code_generator import CodeGenerator

_verbose = 1
def set_log_level(verbose):
    global _verbose
    _verbose = verbose

def get_log_level():
    return _verbose
