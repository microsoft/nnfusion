import code, rlcompleter, readline

"""
This function can start a interactive debug console
example:
from .debug import debug
debug(globals().update(locals()))
"""
def debug(vars):
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    print(globals())
    code.InteractiveConsole(vars).interact()
