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
            # result[0] runtime
            # result[1] error code
            # result[2] compilation time
            if result[1] == 0 and result[0][0] < best_result: # error number equals 0 means no error
                best_result = result[0][0]
                best_step = idx
        return best_result, best_step


def get1(filename):
    best_result = math.inf
    best_step = 0
    current_time = 0
    with open(filename, "r") as f, open(filename[:-4] + ".csv", "w") as of:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            if "r" in obj:
                result = obj["r"]
            else:
                result = obj["result"]
            # result[0] runtime
            # result[1] error code
            # result[2] compilation time
            current_time += result[2]
            # of.write(str(current_time) + "\n")
        print("compilation time:", current_time)
        return best_result, best_step

if __name__ == "__main__":
    filename = sys.argv[1]
    print(get1(filename))