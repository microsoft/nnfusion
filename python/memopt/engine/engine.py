from typing import List
from memopt.graph import Node
from memopt.utils import CompileResult
from .tunner import tune
from memopt import get_log_level

# def run(ordered_nodes, arch):
#     group = {} # map node to fused group
#     dep_count = {node : 0 for node in ordered_nodes}
#     for node in ordered_nodes:
#         for edge in node.outputs:
#             if edge.dst_node in dep_count:
#                 dep_count[edge.dst_node] += 1
#     ready_nodes = list(filter(lambda node:dep_count[node] == 0, dep_count))

#     # cur_node = ordered_nodes[0]
#     for node in ordered_nodes:
#         assert node in ready_nodes
#         if node.is_output():
#             continue
#         for edge in node.outputs:
#             if edge.dst_node in dep_count:
#                 dep_count[edge.dst_node] -= 1
#                 if dep_count[edge.dst_node] == 0:
#                     ready_nodes.append(edge.dst_node)

#         if node not in group:
#             new_group_id = len(set(group.values()))
#             group[node] = new_group_id
#         group_id = group[node]
#         cur_group = list(filter(lambda n: group[n] == group_id, group))

#         ready_output = [set() for _ in node._output_args]
#         unready_output = [set() for _ in node._output_args]
#         for edge in node.outputs:
#             if edge.dst_node in ready_nodes and not edge.dst_node.is_output():
#                 ready_output[edge.src_id].add(edge.dst_node)
#             else:
#                 unready_output[edge.src_id].add(edge.dst_node)

#         for rd, urd in zip(ready_output, unready_output):
#             if len(urd) > 0:
#                 continue
#             else:
#                 new_group = cur_group.copy()
#                 new_group.extend(rd)
#                 print(new_group)
#                 result = tune(new_group, arch, check=True)
#                 if result[0] is not None:
#                     for n in new_group:
#                         group[n] = group_id
#                 print(group)
#     print(group)
#     return group

class FusionGroup():
    def __init__(self, node_list: List[Node], group_id: int, cpresult: CompileResult, gain: float) -> None:
        self.nodes = node_list
        self.group_id = group_id
        self.cpresult = cpresult
        self.gain = gain

def _get_nodes_dependency(nodes, processed):
    # nodes : target nodes in infer dependency
    # processed : already done nodes
    # returns dependency for input nodes (not in processed, not placeholder),
    #          will include input nodes themself.
    queue = list(nodes)
    deps = set()
    while len(queue) > 0:
        node = queue.pop(0)
        deps.add(node)
        for edge in node.inputs:
            if edge.src_node.is_placeholder():
                continue
            if edge.src_node in processed:
                continue
            queue.append(edge.src_node)
    return list(deps)

class Engine:
    def __init__(self, topk: int, arch) -> None:
        self.topk = topk
        self.arch = arch

    def run(self, ordered_nodes: List[Node]) -> List[FusionGroup]:
        self.node2group = {} # map node to fused group
        self.node_topo_id = {ordered_nodes[i] : i for i in range(len(ordered_nodes))}
        fusion_groups = []
        for node in ordered_nodes:
            if node in self.node2group or node.is_output():
                continue
            fg = self._build_fusion_group(node)
            fusion_groups.append(fg)
            if get_log_level() >= 1:
                print("Fusion group created: ", fg.group_id , [node.name for node in fg.nodes])
        return fusion_groups

    def run_no_fusion(self, ordered_nodes: List[Node]) -> List[FusionGroup]:
        fusion_groups = []
        group_id = 0
        for node in ordered_nodes:
            if node.is_output() or node.is_placeholder():
                continue
            result = tune([node], self.arch, node.name, self.topk)
            fusion_groups.append(FusionGroup([node], group_id, result))
            group_id += 1
        return fusion_groups

    def _build_fusion_group(self, top_node):
        cur_group = [top_node]
        cur_group_id = 0 if len(self.node2group) == 0 else max(self.node2group.values()) + 1
        cur_latency_gain = 0
        self.node2group[top_node] = cur_group_id
        queue = [(top_node, i) for i in range(top_node.num_outputs())]
        cp_result = None
        while len(queue) > 0:
            node, output_id = queue.pop(0)
            fusing_nodes = []
            valid = True
            for edge in node.outputs:
                if edge.src_id != output_id:
                    continue
                if edge.dst_node.is_output(): # model output can't be eliminated
                    valid = False
                    break
                if edge.dst_node in fusing_nodes or edge.dst_node in cur_group:
                    continue
                assert edge.dst_node not in self.node2group
                fusing_nodes.append(edge.dst_node)

            if not valid:
                continue

            fusing_nodes = _get_nodes_dependency(fusing_nodes, self.node2group)
            if len(fusing_nodes) == 0 or len(fusing_nodes) > 10: # too many dependency
                continue

            new_group = fusing_nodes + cur_group # create a new subgraph candidate

            # checking group output is valid
            in_group_outputs, out_group_outputs = set(), set()
            for node in new_group:
                for edge in node.outputs:
                    if edge.dst_node in new_group:
                        in_group_outputs.add((node, edge.src_id))
                    else:
                        out_group_outputs.add((node, edge.src_id))
            if in_group_outputs.intersection(out_group_outputs):
                continue

            new_group = sorted(new_group, key=lambda n:self.node_topo_id[n])
            result = tune(new_group, self.arch,
                kernel_name="Group"+str(cur_group_id), topk=self.topk, check=True)
            if result is None:
                continue
            gain = self.compute_gain(new_group, result)
            if gain < cur_latency_gain:
                continue
            cur_latency_gain = gain
            cur_group = new_group
            cp_result = result
            for n in fusing_nodes:
                self.node2group[n] = cur_group_id
                for i in range(n.num_outputs()):
                    queue.append((n, i))

        if cp_result is None: # tune  single op if no fusion is possible
            assert len(cur_group) == 1
            cp_result = tune(cur_group, self.arch,
                kernel_name="Group"+str(cur_group_id), topk=self.topk, check=True)
            if cp_result is None:
                print("Cannot generate code for", top_node)
        return FusionGroup(cur_group, cur_group_id, cp_result, cur_latency_gain)

    def compute_gain(self, group, cp_result):
        for node in group:
            if node.get_tag("latency") is None:
                result = tune([node], self.arch, node.name, self.topk)
                if result is None:
                    latency = 10000
                else:
                    latency = result.latency
                node.add_tag("latency", latency)
        base = sum([node.get_tag("latency") for node in group])
        new = cp_result.latency
        return base - new
