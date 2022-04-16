
import sys
import json

file_name = sys.argv[1]
metrics = ["flop_count_sp_fma", "inst_control"]
items = []

op = "conv"

profile_mode = "profile_time"

def str_to_s(string):
    if string.endswith("ms"):
        return float(string[:-2]) / 1000
    elif string.endswith("us"):
        return float(string[:-2]) / 1000 / 1000
    elif string.endswith("s"):
        return float(string[:-1])

if op == "matmul":
    with open(file_name, "r") as f:
        first = True
        for line in f.readlines():
            if line.startswith("Config:"):
                if first:
                    first = False
                else:
                    items.append(item)
                dic_str = line[8:]
                dic_str = dic_str.replace("\'", "\"")
                config = json.loads(dic_str[:-1])
                reg_x = config['x'][0]
                smem_x = config['x'][1]
                reg_y = config['y'][0]
                smem_y = config['y'][1]
                k = config['k'][0]
                item = [reg_x, reg_y, smem_x, smem_y, k]
            if "default_function_kernel0" in line and "(" in line:
                breaks = line.split()
                t = str_to_s(breaks[1])
                item.append(t)
                for _ in metrics:
                    item.append(0)
            for i in range(len(metrics)):
                metric = metrics[i]
                if metric in line:
                    breaks = line.split()
                    item[6 + i] = float(breaks[-1])

    with open(file_name + ".csv", "w") as ouf:
        ouf.write("reg_x,reg_y,smem_x,smem_y,k,time")
        for metric in metrics:
            ouf.write(",{}".format(metric))
        ouf.write("\n")
        for item in items:
            for i in range(len(item) - 1):
                ouf.write("{},".format(item[i]))
            ouf.write("{}\n".format(item[-1]))

if op == "conv":
    if profile_mode == "profile_metrics":
        has_compute_scale = False
        header = "reg_n,reg_f,reg_h,reg_w,smem_n,smem_f,smem_h,smem_w,rc,time"
        metrics = []
        with open(file_name, "r") as f:
            first = True
            flag = True
            for line in f.readlines():
                if line.startswith("Config: {\'rc\':"):
                    if first:
                        first = False
                    else:
                        items.append(item)
                    import json
                    line = line[8:].replace("\'", "\"")
                    config = json.loads(line[:-1])
                    print(config)
                    reg_n = config["nn"][1]
                    reg_f = config["ff"][1]
                    reg_h = config["xx"][1]
                    reg_w = config["yy"][1]
                    smem_n = config["nn"][0]
                    smem_f = config["ff"][0]
                    smem_h = config["xx"][0]
                    smem_w = config["yy"][0]
                    reg_size = reg_n * reg_f * reg_h * reg_w
                    smem_size = smem_n * smem_f * smem_h * smem_w
                    k = config['rc'][0]
                    grid_size = config["grid_size"]
                    #item = [reg_n, reg_f, reg_h, reg_w, smem_n, smem_f, smem_h, smem_w, k]
                    item = "{},{},{},{},{},{},{},{},{},{}".format(reg_n, reg_f, reg_h, reg_w, smem_n, smem_f, smem_h, smem_w, k, grid_size)
                if "template_op_kernel0" in line and "%" in line:
                    breaks = line.split()
                    t = str_to_s(breaks[-4])
                    item = item + "," + str(t)
                    # insert here so it is only inserted once
                    item = item + "," + str(active_blocks_per_sm)
                    if has_compute_scale:
                        item = item + "," + str(compute_scale)
                    #for _ in metrics:
                    #    item.append(0)
                if line.startswith("Active blocks per SM"):
                    breaks = line.split()
                    active_blocks_per_sm = breaks[-1]
                if line.startswith("Compute Scale"):
                    breaks = line.split()
                    compute_scale = breaks[-1]
                    has_compute_scale = True
                if line.startswith("          1"):
                    breaks = line.split()
                    metric = breaks[1]
                    value = breaks[-1]
                    if len(metrics) > 0 and metrics[0] == metric:
                        flag = False
                    if flag:
                        metrics.append(metric)
                    item = item + "," + value
                """
                for i in range(len(metrics)):
                    metric = metrics[i]
                    if metric in line:
                        print(len(item), i, breaks)
                        item[10 + i] = float(breaks[-1])
                """

        with open(file_name + ".csv", "w") as ouf:
            if has_compute_scale:
                ouf.write("reg_n,reg_f,reg_h,reg_w,smem_n,smem_f,smem_h,smem_w,rc,grid_size,time,active_blocks_per_sm,compute_scale")
            else:
                ouf.write("reg_n,reg_f,reg_h,reg_w,smem_n,smem_f,smem_h,smem_w,rc,grid_size,time,active_blocks_per_sm")
            for metric in metrics:
                ouf.write(",{}".format(metric))
            ouf.write("\n")
            for item in items:
                ouf.write(item+"\n")
                #for i in range(len(item) - 1):
                #    ouf.write("{},".format(item[i]))
                #ouf.write("{}\n".format(item[-1]))

    if profile_mode == "profile_time":
        best_t = -1
        with open(file_name, "r") as f:
            first = True
            flag = True
            for line in f.readlines():
                if line.startswith("Config: {\'rc\':"):
                    if first:
                        first = False
                    else:
                        items.append(item)
                    import json
                    line = line[8:].replace("\'", "\"")
                    config = json.loads(line[:-1])
                    reg_n = config["nn"][1]
                    reg_f = config["ff"][1]
                    reg_h = config["xx"][1]
                    reg_w = config["yy"][1]
                    smem_n = config["nn"][0]
                    smem_f = config["ff"][0]
                    smem_h = config["xx"][0]
                    smem_w = config["yy"][0]
                    reg_size = reg_n * reg_f * reg_h * reg_w
                    smem_size = smem_n * smem_f * smem_h * smem_w
                    k = config['rc'][0]
                    #grid_size = config["grid_size"]
                    #item = [reg_n, reg_f, reg_h, reg_w, smem_n, smem_f, smem_h, smem_w, k]
                    item = "{},{},{},{},{},{},{},{},{}".format(reg_n, reg_f, reg_h, reg_w, smem_n, smem_f, smem_h, smem_w, k)
                breaks = line.split()
                if "%" in line and ("template_op_kernel0" == breaks[-1] or "default_function_kernel0" == breaks[-1]):
                    t = str_to_s(breaks[-4])
                    item = item + "," + str(t)
                    if best_t == -1 or best_t > t:
                        best_t = t
                    #for _ in metrics:
                    #    item.append(0)
                """
                for i in range(len(metrics)):
                    metric = metrics[i]
                    if metric in line:
                        print(len(item), i, breaks)
                        item[10 + i] = float(breaks[-1])
                """


        with open(file_name + ".csv", "w") as ouf:
            ouf.write("reg_n,reg_f,reg_h,reg_w,smem_n,smem_f,smem_h,smem_w,rc,grid_size,time\n")
            for item in items:
                ouf.write(item+"\n")
                #for i in range(len(item) - 1):
                #    ouf.write("{},".format(item[i]))
                #ouf.write("{}\n".format(item[-1]))

        print("{} ms".format(best_t))
