import re

def parse_e2e(filename):
    with open(filename) as f:
        for st in f.readlines():
            st = st.replace(']', "").strip().split()
            if "avg" in st:
                s = st[st.index("avg") + 2]
                return float(s)
            elif "mean" in st:
                s = st[st.index("mean") + 4]
                return float(s)
                

def parse_nvprof(filename):
    with open(filename) as f:
        for st in f.readlines():
            if 'GPU activities:' in st:
                s = st.split()
                rate = float(s[2][:-1])
                tm_str = s[3]
                ss = re.sub("\d*\.\d+","", tm_str)
                if ss == "s":
                    tm = float(tm_str[:-1]) * 1000
                if ss == "ms":
                    tm = float(tm_str[:-2])
                elif ss == "us":
                    tm = float(tm_str[:-2]) / 1000
                elif ss == "ns":
                    tm = float(tm_str[:-2]) / 1000000
                else:
                    raise ValueError("Unknown time unit")
                # print(filename, rate, tm)                
                return tm / rate