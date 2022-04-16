roller_log = "kernel_time_all.dat"
at_log = "runtime_ansor_autotvm.dat"
output_log = "op_perf_multi_v100_raw.dat"

keys = ["Op", "TF(CudaLib)", "TVM", "Ansor", "Roller-Top1", "Roller-Top10"]

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

log1_recs.sort(key = lambda x:x[0])
log2_recs.sort(key = lambda x:x[0])

with open(output_log, "w") as ouf:
    ouf.write("{}\n".format("\t".join(keys)))
    for rec1, rec2 in zip(log1_recs, log2_recs):
        assert rec1[0] == rec2[0]
        ouf.write("{}\n".format("\t".join([rec1[0], rec1[1], rec2[2], rec2[1], rec1[2], rec1[3]])))
