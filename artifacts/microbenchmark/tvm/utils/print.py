import os
import get_best_config
import sys

mypath = sys.argv[1]
files = os.listdir(mypath)
for file in files:
    if file.startswith("our") and file.endswith(".cc"):
        with open(file, "r") as f, open(file+".temp", "w") as of:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx != 2 and idx != 3 and idx != len(lines) - 1:
                    of.write(line)
                    
