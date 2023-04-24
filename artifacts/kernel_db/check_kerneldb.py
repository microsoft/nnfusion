import os

cur_dir = '/home/heheda/.cache/nnfusion'

def content_equal(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        return f1.read() == f2.read()

def get_ids_from(f_path):
    ret = set()
    with open(f_path) as f:
        for line in f:
            ret.add(line.split(":::")[0])    
    return ret

all_ids = get_ids_from("all.id")
ansor_ids = get_ids_from("ansor.id")
roller_ids = get_ids_from("roller.id")
manual_ids = get_ids_from("manual.id")
autotvm_ids = set(["Dot[64,256;3797,256;64,3797floatfloatfloat01]"])
assert all_ids == ansor_ids | roller_ids | manual_ids | autotvm_ids, str(all_ids - (ansor_ids | roller_ids | manual_ids | autotvm_ids)) + str((ansor_ids | roller_ids | manual_ids | autotvm_ids) - all_ids)
assert ansor_ids & roller_ids == set(), str(ansor_ids & roller_ids)
assert ansor_ids & manual_ids == set(), str(ansor_ids & manual_ids)
assert roller_ids & manual_ids == set(), str(roller_ids & manual_ids)
assert ansor_ids & autotvm_ids == set(), str(ansor_ids & autotvm_ids)
assert roller_ids & autotvm_ids == set(), str(roller_ids & autotvm_ids)
assert manual_ids & autotvm_ids == set(), str(manual_ids & autotvm_ids)

cu_equals = set()
json_equals = set()

for root, dirs, files in os.walk(cur_dir):
    if '_db' not in root: continue
    old_root = root.replace("nnfusion", "nnfusion.0411")
    for file in files:
        identifier = file.split(".")[0]
        if identifier not in all_ids:
            for idt in all_ids:
                if idt.startswith(identifier):
                    identifier = idt
                    break
            assert identifier in all_ids, f"{identifier} not in all_ids"
        is_equal = content_equal(os.path.join(root, file), os.path.join(old_root, file))
        # print(f"'{os.path.join(root, file)}'", f"'{os.path.join(old_root, file)}'", is_equal)
        if is_equal:
            if file.endswith(".cu"):
                cu_equals.add(identifier)
            elif file.endswith(".json"):
                json_equals.add(identifier)
            else:
                raise ValueError

with open('all.id') as f:
    for line in f:
        identifier = line.split(":::")[0]
        if not (identifier in cu_equals and identifier in json_equals):
            print(f"Different from paper result: {identifier}:::CUDA_GPU")
