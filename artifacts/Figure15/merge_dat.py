roller_log = "tensor_core_matmul.dat"
at_log = "tensor_core_runtime_autotvm.dat"
output_log = "tensorcore_matmul_v100.dat"

keys = ["OP", "CUBLAS(TF)", "TVM", "Roller-Top1", "Roller-Top10"]

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

with open(output_log, "w") as ouf:
    ouf.write("{}\n".format("\t".join(keys)))
    for rec1, rec2 in zip(log1_recs, log2_recs):
        assert rec1[0] == rec2[0]
        ouf.write("{}\n".format("\t".join([rec1[0], rec1[1], rec2[1], rec1[2], rec1[3]])))
