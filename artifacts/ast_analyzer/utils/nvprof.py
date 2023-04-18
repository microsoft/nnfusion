import ctypes

_cudart = None

def enable_profile(platform):
    if platform == 'V100':
        global _cudart
        _cudart = ctypes.CDLL('libcudart.so')
    elif platform == 'MI100':
        pass # Grinder's experiments don't need profilers on MI100
    else:
        raise NotImplementedError

def profile_start(platform):
    if platform == 'V100':
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)
    elif platform == 'MI100':
        pass
    else:
        raise NotImplementedError


def profile_stop(platform):
    if platform == 'V100':
        ret = _cudart.cudaProfilerStop()
        if ret != 0:
            raise Exception("cudaProfilerStop() returned %d" % ret)
    elif platform == 'MI100':
        pass
    else:
        raise NotImplementedError
