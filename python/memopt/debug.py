import code
import readline
import rlcompleter

"""
This function can start a interactive debug console
example:
from .debug import debug
debug({**globals(), **locals()})
"""
def debug(vars):
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    code.InteractiveConsole(vars).interact()
