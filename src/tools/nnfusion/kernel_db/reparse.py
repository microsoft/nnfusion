import re

func = re.compile(r'(.*__global__\s+void\s+([A-Za-z_]\w*)\s*\(.*\))\s*({.*\Z)')
shared_mem = re.compile(r'__shared__\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;')

def parse(code, parameters):
    for (i, dtype) in enumerate(parameters["dtype"]):