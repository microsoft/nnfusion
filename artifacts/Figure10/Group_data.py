group = [["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C25", "C26", "C27"],
    ["C28", "C29", "C30", "C31", "C34", "C35", "C36", "C37", "C38", "C41", "C42", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"],
    ["D15", "D17", "D18", "D19", "D21", "D22", "E1", "E2", "E3", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21"],
    ["E22", "E23", "E24", "E25", "E26", "E27", "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "R0", "R1", "R2", "R3", "M0", "M1", "M2"],
    ["C10", "C24", "C32", "C33", "C39", "C40", "C43", "D16", "D20", "E0", "E4", "M3", "M4", "M5", "M6"]]
keys = ["OP", "TF", "TVM", "Ansor", "Roller-Top1", "Roller-Top10"]
group_size = 28

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

log_keys, log_recs = parse_dat("op_perf_multi_v100_raw.dat")

with open("op_perf_multi_v100.dat", "w") as ouf:
    for i in range(5):
        ouf.write("\t".join(keys))
        if i < 4:
            ouf.write("\t")
        else:
            ouf.write("\n")
    for l in range(group_size):
        for g in range(5):
            if l >= len(group[g]):
                continue
            for rec in log_recs:
                if rec[0] == group[g][l]:
                    ouf.write("\t".join(rec))
                    ouf.write('\t')
                    break
        ouf.write("\n")
