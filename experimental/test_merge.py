
import welder
from welder.arch import *
from welder.engine import Tunner, load_model

welder.set_log_level(3)
ordered_nodes = load_model("/home/v-yiningshi/learn_tvm/testing/temp/Conformer/1_fp16/model.json")

l = ['Broadcast_Add_Sum_4581', 'Reshape_Add_4603']
# l = ['Sum_Broadcast_Divide_1837', 'Reshape_Reshape_Broadcast_Subtract_Multiply_1856']
nodes = []
for node in ordered_nodes:
    if node.name in l:
        nodes.append(node)


# best = Tunner(V100(), device=0, topk=20).tune(nodes)

best = Tunner(V100(), device=0, topk=20).tune(nodes, [[nodes[0].name, nodes[1].name]])
value0 = best.get_example_outputs()[0]
print(best.code)

best = Tunner(V100(), device=0, topk=20).tune(nodes)
value1 = best.get_example_outputs()[0]
import numpy as np

print(value0)
print("_" * 100)
print(value1)
print(np.max(np.abs(value0 - value1)))

# config = Config().from_dict({'block': [128, 192], 'warp': [64, 96], 'wmma': [32, 32, 4], 'use_cutlass': True, 'rstep': [64], 'use_tc': '70'})

# config.tc_extra_conf = TensorCoreExtraConfig(None, None, None, None, [0, 1, 0, 1, 0, 1])
# cgen = welder.CodeGenerator()
# cpresult = cgen.compile(output_nodes, {output_nodes[0].inputs[0].src_node: config}, "cuda", "Fused")
# cpresult.compile_and_load(V100())
# for i in range(10):
#     print(cpresult.profile())
# best = Tunner(V100(), device=0, topk=20).tune(nodes)
# print(best.code)
