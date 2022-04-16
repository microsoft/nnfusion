import re

def parse_launch_config(source):
	rule = "attr \[IterVar\((\w+\.\w+): int32, \(nullptr\), \"ThreadIndex\", \"(\w+\.\w+)\"\)\] \"thread_extent\" = (\d+)"
	res = re.findall(rule, source)
	size = {
		"blockIdx.x": 1, 
		"blockIdx.y": 1,
		"blockIdx.z": 1,
		"threadIdx.x": 1,
		"threadIdx.y": 1,
		"threadIdx.z": 1
	}
	for r in res:
		if r[0] == r[1]:
			size[r[0]] = int(r[2])
	return (size["blockIdx.x"], size["blockIdx.y"], size["blockIdx.z"]), (size["threadIdx.x"], size["threadIdx.y"], size["threadIdx.z"])