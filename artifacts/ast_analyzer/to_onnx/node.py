class OnnxNodes:
    def __init__(self, def_nodes=None, out_node=None, def_value_infos=None):
        # type: Optional[List[NodeProto]], Optional[List[str]], Optional[Map[str, ValueInfoProto]]
        self.def_nodes = def_nodes if def_nodes is not None else []
        self.out_node = out_node if out_node is not None else []
        self.def_value_infos = def_value_infos if def_value_infos is not None else {}

    def __add__(self, node):
        # type: OnnxNodes -> OnnxNodes
        assert(isinstance(node, OnnxNodes))
        return OnnxNodes(self.def_nodes + node.def_nodes, self.out_node + node.out_node, {**self.def_value_infos, **node.def_value_infos})

    def set_output(self, node, name, value_info=None):
        # type: ASTNode, str, ValueInfoProto -> OnnxNodes
        self.def_nodes.append(node)
        self.out_node = [name]
        if value_info is not None:
            self.def_value_infos[name] = value_info

    def set_outputs(self, nodes, names, value_infos=None):
        # type: ASTNode, List[str], Map[str, ValueInfoProto]
        if nodes is not None:
            self.def_nodes.extend(nodes)
        self.out_node = names
        if value_infos is not None:
            self.def_value_infos.update(value_infos)

    def out_is_tensor(self):
        # type: * -> None (out_name not in value_info) / True (out is tensor) / False (out is not tensor)
        assert(len(self.out_node) == 1)
        name = self.out_node[0]
        if name not in self.def_value_infos:
            return None
        value_info = self.def_value_infos[name]
        return value_info.type.HasField('tensor_type')

    def __str__(self):
        # type: * -> str
        return "[defs: {}, outs: {}, value_infos: {}]".format(
            ','.join([x.op_type for x in self.def_nodes]),
            ','.join([x for x in self.out_node]),
            ','.join([x.name for x in self.def_value_infos.values()])
        )
