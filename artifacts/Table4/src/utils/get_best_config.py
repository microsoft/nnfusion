import json
import sys
import math

def get(filename):
    best_result = math.inf
    best_step = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            if "r" in obj:
                result = obj["r"]
            else:
                result = obj["result"]
            if result[1] == 0 and result[0][0] < best_result: # error number equals 0 means no error
                best_result = result[0][0]
                best_step = idx
        return best_result, best_step

if __name__ == "__main__":
    filename = sys.argv[1]
    print(get(filename))