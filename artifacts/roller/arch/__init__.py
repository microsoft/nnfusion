from .k80 import *
from .ipu import *
from .v100 import *
from .mi50 import *
from .rtx3090 import *
from .xbox import *
from .xbox_xdx import *


def dispatch_arch(device_name):
  if device_name == 'K80':
    return K80()
  elif device_name == 'IPU':
    return IPU()
  elif device_name == 'V100':
    return V100()
  elif device_name == 'MI50':
    return MI50()
  elif device_name == 'RTX3090':
    return RTX3090()
  elif device_name == 'Xbox':
    return Xbox()
  elif device_name == 'XboxSeriesXDX':
    return XboxSeriesXDX()
  else:
    assert False, 'ERROR: {} NOT supported'.format(device_name)
