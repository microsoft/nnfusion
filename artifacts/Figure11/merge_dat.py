roller_log = "compile_time_roller.dat"
at_log = "compile_time_ansor_autotvm.dat"
output_log = "op_compile_time_v100.dat"

keys = ["OP", "TVM", "Ansor", "Roller-Top1", "Roller-Top10"]

log1_keys = []
log1_recs = []
log2_keys = []
log2_recs = []

def parse_dat(log_name):
    first = True
    log_keys = []
    log_recs = []
    with open(log_name, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if first:
                log1_keys = line.split("\t")
                first = False
            else:
                log_recs.append(line.split("\t"))
    return log_keys, log_recs

log1_keys, log1_recs = parse_dat(roller_log)
log2_keys, log2_recs = parse_dat(at_log)

roller1_time = [float(x[1]) for x in log1_recs]
roller10_time = [float(x[2]) for x in log1_recs]
ansor_time = [float(x[1]) for x in log2_recs]
autotvm_time = [float(x[2]) for x in log2_recs]

roller1_time.sort()
roller10_time.sort()
ansor_time.sort()
autotvm_time.sort()

with open(output_log, "w") as ouf:
    ouf.write("{}\n".format("\t".join(keys)))
    for i in range(len(roller1_time)):
        ouf.write("{}\t{}\t{}\t{}\t{}\n".format(i, autotvm_time[i], ansor_time[i], roller1_time[i], roller10_time[i]))
