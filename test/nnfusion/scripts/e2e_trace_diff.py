# Microsoft (c) 2019, MSRA/NNFUSION Team
# Author: wenxh
# This script is to be used to diff the trace
import os
import re
import sys
import subprocess
import logging
import numpy as np

if len(sys.argv) != 3:
    logging.error("Script doesn't have right arguments.")
    logging.error(
        "python e2e_trace_diff.py nnfusion_debug_trace.txt tf_debug_trace.txt")
    exit(1)


class node:
    def __init__(self, name):
        self.name = name
        self.inputs = list()
        self.outputs = list()
        self.map_node = ""  # "node" : 1.0(confidence)
        self.confident_list = list()
        self.output_data = None

    def add_children(self, children):
        self.outputs.append(children)
        return self

    def add_parent(self, parent):
        self.inputs += parent
        return self

    def add_data(self, data):
        self.output_data = np.array(data, dtype=float)


class tracefile:
    def __init__(self):
        self.allnodes = dict()
        self.entries = list()
        self.outputs = list()
        self.match = set()
        self.boundary_visited = set()
        self.boundary = set()
        self.cacheallclose = dict()
        self.ischild = dict()
        self.cache_subgraph_match = dict()

    def read_nnfusion_trace(self, file):
        #  node: 0.0  0.0 : input1, input2
        f = open(file).readlines()
        for line in f:
            line = line.strip()
            if line.endswith(":"):
                break
            segs = line.split(":")

            name = segs[0].strip()
            inputs = [v.strip() for v in segs[3].split(",")]
            data1 = [float(v.strip()) for v in segs[1].strip().split(
                "...(")[0].strip().split(" ")]
            data2 = [float(v.strip()) for v in segs[2].strip().split(
                "...(")[0].strip().split(" ")]
            data = data1 + data2

            mapnode = ""
            if "," in name:
                mapnode = name.split(",")[0].strip()
                name = name.split(",")[1].strip()
            # create node
            n = node(name)
            n.add_parent(inputs)
            n.add_data(data)
            self.allnodes[name] = n
            self.allnodes[name].map_node = mapnode

            # add children
            for p in inputs:
                if p not in self.allnodes.keys():
                    self.allnodes[p] = node(p)
                    self.entries.append(name)
                self.allnodes[p].add_children(name)

            logging.info(("%s <- %s : %s") % (name, inputs, data))

        for nname in self.allnodes.keys():
            if len(self.allnodes[nname].inputs) == 0:
                self.entries.append(nname)
            if len(self.allnodes[nname].outputs) == 0:
                self.outputs.append(nname)

    def read_tf_trace(self, file):
        f = open(file)
        while True:
            # dense_1/bias:0 <- xxx:0, xxx:0
            # [0. 0. 0. 0. 0.
            # 0. 0. 0. 0. 0.] ...(size= 512 end with 0.0 )
            line = f.readline()
            if line:
                if "<-" not in line:
                    break
                ls = line.split("<-")
                name = ls[0].strip()
                inputs = [v.strip() for v in ls[1].strip().split(",")]
                n = node(name)
                n.add_parent(inputs)
                self.allnodes[name] = n

                # add children
                for p in inputs:
                    if p not in self.allnodes.keys():
                        self.allnodes[p] = node(p)
                    self.allnodes[p].add_children(name)

                line = ""
                while "] ...(size=" not in line:
                    line += f.readline()
                # hard fix for bool
                line = line.replace("[False]", "[0]")
                data1 = [float(v) for v in re.split(
                    "\s+", line.split("]")[0][1:].strip())]

                line = ""
                while "] offset= " not in line:
                    line += f.readline()
                # hard fix for bool
                line = line.replace("[False]", "[0]")
                data2 = [float(v) for v in re.split(
                    "\s+", line.split("]")[0][1:].strip())]
                data = data1 + data2

                self.allnodes[name].add_data(data)

                logging.info(("%s <- %s : %s") % (name, inputs, data))
            else:
                break

        for nname in self.allnodes.keys():
            if len(self.allnodes[nname].inputs) == 0:
                self.entries.append(nname)
            if len(self.allnodes[nname].outputs) == 0:
                self.outputs.append(nname)

    def rouge_match(self, trace):
        for v in self.allnodes.keys():
            flag = False
            for nnfname in self.allnodes.keys():
                if nnfname.startswith(self.allnodes[v].map_node):
                    flag = True
                    break
            if len(self.allnodes[v].map_node) > 0 and not flag:
                for tfname in trace.allnodes.keys():
                    if tfname[:-2] == self.allnodes[v].map_node:
                        if self.allclose(v, trace, tfname):
                            self.allnodes[v].confident_list.append(tfname)
                            self.match.add(v)
            else:
                for u in trace.allnodes.keys():
                    if self.allclose(v, trace, u):
                        self.allnodes[v].confident_list.append(u)
        
        for f in trace.allnodes.keys():
            for s in trace.allnodes.keys():
                trace.is_child(f, s)

    def add_valid_parent(self, root):
        for parent in self.allnodes[root].inputs:
            if parent not in self.match:
                self.match.add(parent)
                self.add_valid_parent(parent)
    
    def allclose(self, a, trace, b):
        if a+b not in self.cacheallclose.keys():
            if (not self.allnodes[a].output_data is None)\
                and (not trace.allnodes[b].output_data is None)\
                and len(self.allnodes[a].output_data) == len(trace.allnodes[b].output_data)\
                and np.allclose(self.allnodes[a].output_data, trace.allnodes[b].output_data, rtol=1.e-4, atol=1.e-4):
                self.cacheallclose[a+b] = True
            else:
                self.cacheallclose[a+b] = False
        return self.cacheallclose[a+b]
    
    def is_child(self, father, son):
        if father == "__root__":
            return True
        if (father+son) in self.ischild.keys():
            return self.ischild[father+son]
        if father == son:
            return True
        start_node = self.allnodes[father]
        for u in start_node.outputs:
            if self.is_child(u, son):
                self.ischild[u + son] = True
                return True
        self.ischild[father + son] = False
        return False

    def subgraph_match(self, cur_node, trace, trace_node, dep=0):
        if cur_node + trace_node in self.cache_subgraph_match.keys():
            return self.cache_subgraph_match[cur_node + trace_node]

        tabs = "".join(["-"]*dep)

        if self.allclose(cur_node, trace, trace_node):
            logging.info("%s%s --allclose--> %s" %
                         (tabs, cur_node, trace_node))
            if cur_node in self.outputs:
                self.cache_subgraph_match[cur_node+trace_node] = True
                logging.info(
                    "%s^------- Confident match path ends here." % (" "*dep))
                return True

        ret_flag = True
        for subnode in self.allnodes[cur_node].outputs:
            node_flag = False
            # valid for one case
            for trace_sub_node in self.allnodes[subnode].confident_list:
                if trace.is_child(trace_node, trace_sub_node):
                    if self.subgraph_match(subnode, trace, trace_sub_node, dep + 1):
                        node_flag = True

            sub_flag = False
            no_sub = True
            for tfname in self.allnodes[subnode].confident_list:
                if tfname[:-2] == self.allnodes[subnode].map_node:
                    # cannot skip this node
                    no_sub = False
            if no_sub:
                sub_flag = len(self.allnodes[subnode].outputs) > 0
                for subsubnode in self.allnodes[subnode].outputs:
                    if not self.subgraph_match(subsubnode, trace, trace_node, dep + 1):
                        sub_flag = False
                        break

            ret_flag = ret_flag and (node_flag or sub_flag)

        self.cache_subgraph_match[cur_node+trace_node] = ret_flag
        return ret_flag

    
    def find_boundary(self, root, trace, trace_node):
        if root in self.boundary_visited:
            return
        self.boundary_visited.add(root)

        if root not in self.match:
            return

        for sub in self.allnodes[root].outputs:
            if sub not in self.match:
                for sub_trace_node in self.allnodes[sub].confident_list:
                    if trace.is_child(trace_node, sub_trace_node):
                        if self.subgraph_match(sub, trace, sub_trace_node):
                            self.match.add(sub)

        for sub in self.allnodes[root].outputs:
            if sub not in self.match:
                self.boundary.add(sub)
            else:
                sub_trace_node = trace_node
                if len(self.allnodes[sub].confident_list) == 1:
                    trace_node = self.allnodes[sub].confident_list[0]
                self.find_boundary(sub, trace, sub_trace_node)

    def compare_with(self, trace):
        self.rouge_match(trace)
        initlist = list(self.match)
        for valid in initlist:
            self.add_valid_parent(valid)
        
        for n in self.entries:
            trace_node = "__root__"
            if len(self.allnodes[n].confident_list) == 1:
                trace_node = self.allnodes[n].confident_list[0]
            self.find_boundary(n, trace, trace_node)
        
        print("[Error Boundary] " + ", ".join(self.boundary))
        return False


nnf_trace = tracefile()
nnf_trace.read_nnfusion_trace(sys.argv[1])
tf_trace = tracefile()
tf_trace.read_tf_trace(sys.argv[2])

if nnf_trace.compare_with(tf_trace):
    exit(0)
else:
    exit(1)
