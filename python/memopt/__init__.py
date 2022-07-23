import torch
from .IRpass import *
from .scope import get_scope, Scope
from .schedule_rewrite import CodeGenerator
from . import utils

_verbose = 1
def set_log_level(verbose):
    global _verbose
    _verbose = verbose

def get_log_level():
    return _verbose
