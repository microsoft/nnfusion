
from welder.arch import *
from welder.engine import Tunner, load_model

ordered_nodes = load_model("example.json")

def get_nodes(l):
    nodes = []
    for node in ordered_nodes:
        if node.name in l:
            nodes.append(node)
    return nodes

l1 = ["BatchMatMul_Broadcast_Divide_1", "Reshape_Broadcast_Add_2"]
l2 =  ["SoftmaxBasic_3", "SoftmaxBasic_4", "SoftmaxBasic_5", "SoftmaxBasic_6"]

tunner = Tunner(V100(), device=0, topk=20)
unfused1 = tunner.tune(get_nodes(l1))
print("Matmul unfused latency: ", unfused1.latency)
unfused2 = tunner.tune(get_nodes(l2))
print("Softmax unfused latency: ", unfused2.latency)
total = tunner.tune(get_nodes(l1 + l2))
print("Fused op latency: ", total.latency)
