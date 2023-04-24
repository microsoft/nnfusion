import numpy as np
import re

sys_name = "Cocktailer"

line_markers = [
    'x',
    '^',
    '*',
    'o'
]

line_styles = [
    'dotted',
    'dashed',
    'solid'
]

line_colors = [
    '#fccde5',
    '#ffd93c',
    '#4169e1',
    '#6fd7ee',
    '#f4684d',
    # '#a52a2a'
]
    # '#73e069',

color_def = [
    '#f4b183',
    '#c5e0b4',
    '#ffd966',
    '#bdd7ee',
    "#fb8072",
    # "#8dd3c7",
    # "#bebada",
]

plot_dir = '../reproduce_results/plot'

SHOW=False

def geo_mean(mat):
    return np.exp(np.log(np.array(mat)).mean())

def parse_time(f_path):
    pattern = re.compile(r"min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, avg = (\d+\.\d+) ms")
    with open(f_path) as f:
        for line in f:
            if "min" in line and "max" in line and "avg" in line:
                t = float(pattern.search(line).group(3))
                return t
    return None

def parse_tf_time(f_path):
    # from a line like:
    # [31mSummary: [min, max, mean] = [5.148649, 5.614281, 5.427058] ms[m
    pattern = re.compile(r"\[min, max, mean\] = \[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\] ms")
    with open(f_path) as f:
        for line in f:
            if "min" in line and "max" in line and "mean" in line:
                t = float(pattern.search(line).group(3))
                return t
    return None

def parse_kernel_time(f_name):
    with open(f_name) as f:
        lines = f.readlines()
        pattern = re.compile('([\d\.]+)([a-z]+)')
        for line in lines:
            if 'GPU activities' in line:
                rate = float(line.split()[2][:-1])
                kernel_time_str = line.split()[3]
                m = pattern.match(kernel_time_str)
                t = float(m.group(1))
                unit = m.group(2)
                if unit == 's':
                    t *= 1000
                elif unit == 'ms':
                    pass
                elif unit == 'us':
                    t /= 1000
                elif unit == 'ns':
                    t /= 1000000
                else:
                    raise ValueError("Unknown unit: " + unit)
                return t / rate
    return None
