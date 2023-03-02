import onnx
import copy
import os
import argparse
from onnx import helper


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./model.onnx', help='The file name of the frozen graph.')
parser.add_argument('--mp_file', type=str, default='model.onnx', help='')

args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

inedge_node_map = {}

def pattern_match(node, pattern, idx, matched, result):
    if node.op_type == pattern[idx - 1] and idx == len(pattern):
        result.append(matched[:])
        return
    for output in node.output:
        assert output in inedge_node_map
        subnode = inedge_node_map[output]
        if subnode.op_type == pattern[idx]:
            matched.append(subnode)
            pattern_match(subnode, pattern, idx + 1, matched, result)
            matched.pop()
        

def add_cast(node, cast_input, cast_output):
    replace_nodes = []
    # cast input to fp32
    if cast_input:
        fp32_input = []
        for input in node.input:
            fp32_name = input + ".fp32"
            cast_op = helper.make_node(
                "Cast",
                [input],
                [fp32_name],
                to=1,
            )
            replace_nodes.append(cast_op)
            fp32_input.append(fp32_name)            
        node.ClearField('input')
        node.input.extend(fp32_input)

    replace_nodes.append(node)
    
    # cast output to fp16
    if cast_output:
        fp16_output = []
        for output in node.output:
            fp16_name = output + ".fp16"
            # note: here we cast output to fp16 regardless of original input type
            cast_op = helper.make_node(
                "Cast",
                [fp16_name],
                [output],
                to=10,
            )
            replace_nodes.append(cast_op)
            fp16_output.append(fp16_name)
        node.ClearField('output')
        node.output.extend(fp16_output)
    return replace_nodes

def main():
    model  = onnx.load(args.file)
    fp32_patterns = [["Pow", "ReduceMean"], ["ReduceMean"], ["Softmax"], ["InstanceNormalization"]]
    origin_nodes = copy.deepcopy(model.graph.node)
    model.graph.ClearField('node')
    for node in origin_nodes:
        for input in node.input:
            inedge_node_map[input] = node
    visited = set()
    for node in origin_nodes:
        if node.name in visited:
            continue
        visited.add(node.name)
        in_pattern = False
        for pattern in fp32_patterns:
            if node.op_type == pattern[0]:
                result = []
                pattern_match(node, pattern, 1, [node], result)
                matched = result[0]
                if len(matched) == len(pattern):
                    for m in matched:
                        visited.add(m.name)
                    if len(matched) == 1:
                        replace_nodes = add_cast(matched[0], True, True)
                        model.graph.node.extend(replace_nodes)
                    else:
                        replace_nodes_in = add_cast(matched[0],True, False)
                        model.graph.node.extend(replace_nodes_in)
                        model.graph.node.extend(matched[1:-1])
                        replace_nodes_out = add_cast(matched[-1],False, True)
                        model.graph.node.extend(replace_nodes_out)
                    in_pattern = True
                    break
        if not in_pattern:
            model.graph.node.extend([node])
    onnx.save(model, args.mp_file, save_as_external_data=True, all_tensors_to_one_file=True)

if __name__ == "__main__":
    main()