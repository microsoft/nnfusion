import pandas as pd

read_metric_name = "gld_throughput"
write_metric_name = "gst_throughput"

metrics = pd.read_csv("./metrics", skiprows=5)
prof = pd.read_csv("./profile", skiprows=3)

flops = metrics[metrics['Metric Name'] == "flop_count_sp"]
total_flops_sp = sum(flops["Invocations"].astype("int64") * flops["Avg"].astype("int64"))

kernel_to_time = {}
for _, row in prof.iterrows():
    if type(row["Type"]) is float:
        time_elm = row["Time"]
    if row["Type"] != "GPU activities": continue
    kernel_to_time[row["Name"]] = float(row["Time"]) * {"us":1e-6, "ms":1e-3}[time_elm]
totel_exec_time = sum(kernel_to_time.values())

gld_bytes, gst_bytes = 0, 0
for _, row in metrics.iterrows():
    if row['Metric Name'] == read_metric_name:
        throughput = row["Avg"]
        value = float(throughput[:-4]) * {'G':1e9, 'M':1e6, 'K':1e3, '0': 1}[throughput[-4]]
        gld_bytes += kernel_to_time[row["Kernel"]] * value
    elif row['Metric Name'] == write_metric_name:
        throughput = row["Avg"]
        value = float(throughput[:-4]) * {'G':1e9, 'M':1e6, 'K':1e3, '0': 1}[throughput[-4]]
        gst_bytes += kernel_to_time[row["Kernel"]] * value

print("Total GLD: {} M Transaction, GST: {} M Transaction, TOT: {} M Transaction".format(
    gld_bytes / 1e6 / 32, gst_bytes / 1e6 / 32, (gld_bytes + gst_bytes) / 1e6 / 32))
