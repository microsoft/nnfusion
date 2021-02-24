# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os, sys, subprocess, gzip
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.client import timeline
from google.protobuf import text_format, json_format
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.tools import graph_transforms
from typing import List
from convert_graph_fp16 import* 

class nnf_tf_freezer(object):
    def __init__(self, frozen_graph= "frozen_graph.pb", const_folding=True, run_graph=True, xla=False, parallel=0,
    warmup=5, num_iter=10, run_const_folded_graph=False, debug=False, is_training=False, to_fp16=False):
        self.frozen_graph = frozen_graph
        self.const_folding = const_folding
        self.run_graph = run_graph  
        self.xla = xla 
        self.parallel = parallel
        self.warmup = warmup
        self.num_iter = num_iter
        self.folded_graph = None
        self.run_const_folded_graph = run_const_folded_graph
        self.debug = debug
        self.is_training = is_training
        self.to_fp16 = to_fp16

    def execute(self, inputs : List[tf.placeholder], outputs : List[tf.identity], optimizer : tf.train.Optimizer=None):      
        self.freeze(inputs, outputs, optimizer)
        if self.const_folding:
            self.tf_run_const_folding(self.frozen_graph)
        if self.run_graph:
            if self.folded_graph and self.run_const_folded_graph:
                print('run constant-folded graph: ')
                self.tf_run_frozen_graph(self.folded_graph, self.xla, self.parallel, self.warmup, self.num_iter)
            else:
                print('run original frozen graph: ')
                self.tf_run_frozen_graph(self.frozen_graph, self.xla, self.parallel, self.warmup, self.num_iter)

    def freeze(self, inputs : List[tf.placeholder], outputs : List[tf.identity], optimizer : tf.train.Optimizer=None):
        print("Freeze graph ----------------------------------") 
        varlist = []             
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.is_training:
                logits = outputs[0] # assume outputs[0] is logits
                labels = tf.placeholder(tf.int32, shape=[logits.shape[0], ], name="nnfusion/labels")
                inputs += [labels]
                if self.const_folding:
                    loss = tf.identity(tf.identity(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits), name="nnfusion/loss"), name="nnfusion/loss_identity")
                else:
                    loss = tf.identity(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits), name="nnfusion/loss")
                outputs += [loss]
                if optimizer:
                    opt = optimizer
                else:
                    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
                grads = opt.compute_gradients(loss)
                train_op = opt.apply_gradients(grads)

                sess.run(tf.global_variables_initializer())
                for t in train_op.control_inputs:
                    for ot in t.outputs:
                        outputs += [tf.identity(ot, name = 'nnfusion_grads/' + ot.name.split(':')[0])]
                    # outputs += grads
                    for op in sess.graph_def.node:
                        if "Apply" in str(op.op) or "Assign" in str(op.op) or "Scatter" in str(op.op):
                            varlist.append(op.input[0])
                varlist = ",".join(varlist)
                if self.debug:
                    print(varlist) 
            tf.train.write_graph(sess.graph_def, '/tmp/save', 'model.pbtxt')
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            try:
                saver.restore(sess, "/tmp/save/model.ckpt")
                print('Using existing checkpoint.')
            except:
                print('Not using existing checkpoint.')
                pass

            saver_path = saver.save(sess, "/tmp/save/model.ckpt")
            
            if self.to_fp16:
                # convert graph to fp16 model
                print('convert to fp16 model')
                input_name = [input.name for input in inputs]
                output_names = [output.name for output in outputs]
                
                new_graph = convert_graph_to_fp16(sess.graph_def, target_type='fp16', input_name=input_name, output_names=output_names)
                tf.train.write_graph(new_graph, '/tmp/save', 'model.pbtxt')

            freeze_graph.freeze_graph(
                input_graph="/tmp/save/model.pbtxt",
                input_checkpoint="/tmp/save/model.ckpt",
                output_node_names=','.join([x.name.split(':')[0] for x in outputs]),
                output_graph= self.frozen_graph,
                input_saver="", 
                input_binary=False, 
                restore_op_name='save/restore_all', 
                filename_tensor_name='save/Const:0', 
                clear_devices=True, 
                initializer_nodes="", 
                variable_names_blacklist = varlist)
            '''
            self.graphdef_to_json(self.frozen_graph, self.frozen_graph + ".json.gz")
            ops_used = subprocess.getoutput("zgrep -v tensorContent " + self.frozen_graph + ".json.gz | grep '\"op\":' | sort | uniq | awk -F'\"' '{print $4}' | xargs echo").split()
            os.system('zgrep -v tensorContent ' + self.frozen_graph + '.json.gz > ' + self.frozen_graph + '.json.thin')
            print('>> Ops used by Graph `%s`:' % self.frozen_graph)
            print(ops_used)
            with open(self.frozen_graph + '.ops', 'w') as fp:
                for op in ops_used:
                    fp.write(op + '\n')
            '''

    def tf_run_const_folding(self, file):
        print("run const folding----------------------------")
        tf.reset_default_graph()
        graph_def, graph = self.import_graph(file)
            
        print()
        if (self.debug):
            print('Placeholders:')
        assert graph is not None
        ops = graph.get_operations()  # type: Iterable[tf.Operation]
        input_nodes = []
        last_nodes = []
        for op in ops:
            if op.type == 'Placeholder':
                for tensor in op.outputs:
                    if (self.debug):
                        print('- {0:20s} {1}'.format("Tensor", tensor.name))
                    input_nodes.append(tensor.name)
        
        if (self.debug):            
            print()
            print('Sinks (operations without outputs):')
        last_outputs = []
        num_nodes = len(ops)
        name2nodeIdx_map = {}
        for i in range(num_nodes):
            name2nodeIdx_map[ops[i].name] = i
        node_outputs_ = [[] for i in range(num_nodes)]
        for n in range(num_nodes):
            op = ops[n]
            pending_count = len(op.inputs)
            for i in range(pending_count):
                input_name_id = op.inputs[i].name.split(':')
                node_outputs_[name2nodeIdx_map[input_name_id[0]]].append(n)
        for n in range(num_nodes):
            if len(node_outputs_[n]) == 0 and ops[n].type != 'NoOp':
                if (self.debug):
                    print('- {0:20s} {1}'.format(ops[n].type, ops[n].name))
                for m in range(len(ops[n].inputs)):
                    if (self.debug):
                        print('<-in-- {0:20s}'.format(ops[n].inputs[m].name))
                    last_outputs.append(ops[n].inputs[m].name)
            '''
            if len(node_outputs_[n]) == 0 and ops[n].type == 'NoOp':
                for m in range(len(ops[n].control_inputs)):
                    print('<-in-^ {0:20s}'.format(ops[n].control_inputs[m].name))
                    last_outputs.append(ops[n].control_inputs[m].name)
            '''
        print(input_nodes)
        print(last_outputs)
        g_def_const = tf.import_graph_def(graph_def, name="")
        g_def_const = graph_transforms.TransformGraph(graph_def, input_nodes, last_outputs, ["fold_constants", "strip_unused_nodes"])

        print()
        self.folded_graph = file[:-3] + ".const_folded.pb"
        print("Saving Const-folded Graph... as " + self.folded_graph)
        graph_io.write_graph(as_text=False, name=self.folded_graph, logdir="./",graph_or_graph_def=g_def_const)
        print("Finished.")

    def tf_run_frozen_graph(self, file, xla, parallel, warmup, num_iter):
        print("run frozen graph----------------------------")
        graph_def, graph = self.import_graph(file)
        if (self.debug):
            print()
            print('Operations:')
        assert graph is not None
        ops = graph.get_operations()  # type: Iterable[tf.Operation]
        input_nodes = []
        variables_nodes = []
        last_nodes = []
        for op in ops:
            if (self.debug):
                print('- {0:20s} "{1}" ({2} outputs)'.format(op.type, op.name, len(op.outputs)))
            last_nodes = op.outputs
            if op.type == 'Placeholder':
                for node in op.outputs:
                    input_nodes.append(node)
            if "Variable" in op.type:
                variables_nodes.append(op)

        if (self.debug):
            print()
            print('Sources (operations without inputs):')
            for op in ops:
                if len(op.inputs) > 0:
                    continue
                print('- {0}'.format(op.name))

            print()
            print('Operation inputs:')
            for op in ops:
                if len(op.inputs) == 0:
                    continue
                print('- {0:20}'.format(op.name))
                print('  {0}'.format(', '.join(i.name for i in op.inputs)))

            print()
            print('Tensors:')
            for op in ops:
                for out in op.outputs:
                    print('- {0:20} {1:10} "{2}"'.format(str(out.shape),
                                                        out.dtype.name, out.name))
        with tf.Session(graph=graph) as sess:
            var_inits = []
            g_def = graph.as_graph_def()
            for var in variables_nodes:
                vt = graph.get_tensor_by_name(var.outputs[0].name)
                # v = tf.get_variable(name = var.name, shape = vt.shape, initializer = tf.ones_initializer)
                # v = tf.get_variable(name = var.name, shape = vt.shape, initializer = tf.ones_initializer)
                # Ones initializer
                dt = tf.as_dtype(vt.dtype.base_dtype).as_datatype_enum
                dt_int32 = tf.as_dtype(tf.int32).as_datatype_enum 

                init = tf.NodeDef(
                    name = var.name + "/ones",
                    op = "Fill",
                    input = [var.name + "/ones/shape", var.name + "/ones/const"],
                    attr = {
                        'T': tf.AttrValue(type=dt),
                        'index_type': tf.AttrValue(type=dt_int32)
                    }
                )

                shape = tf.NodeDef(
                    name = var.name + "/ones/shape",
                    op = "Const",
                    attr = {
                        "dtype": tf.AttrValue(type=dt_int32),
                        "value": tf.AttrValue(tensor = tf.make_tensor_proto(vt.get_shape().as_list()))
                    }
                )

                const = tf.NodeDef(
                    name = var.name + "/ones/const",
                    op = "Const",
                    #dtype =tf.AttrValue(type=dt),
                    attr = {
                        "dtype": tf.AttrValue(type=dt),
                        "value": tf.AttrValue(tensor = tf.make_tensor_proto(1.0, dt))
                    }
                )

                node = tf.NodeDef( name=var.name + "/assign", op='Assign',input=[var.name, var.name+"/ones"], 
                    attr={'use_locking': tf.AttrValue(b=False), 'validate_shape': tf.AttrValue(b=True),
                    'T': tf.AttrValue(type=dt)})
                g_def.node.extend([shape, const, init, node])
                var_inits.append("^" + var.name + "/assign")
            
            noop_assign = tf.NodeDef(name = "init_all_var", op="NoOp", input = var_inits)
            g_def.node.extend([noop_assign])

        tf.reset_default_graph()
        tf.import_graph_def(g_def)

        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=parallel
        )

        if xla:
            session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=session_conf) as sess:
            init = tf.get_default_graph().get_operation_by_name("import/init_all_var")

            input_nodes = []
            varlist = []
            feed_dict = {}
            aps = []

            ops = tf.get_default_graph().get_operations()
            for op in ops:
                if op.type == 'Placeholder':
                    for node in op.outputs:
                        feed_dict[node] = np.ones(node.shape, dtype=node.dtype.as_numpy_dtype())

            # Get result of applygradient
            for op in ops:
                if "ApplyGradient" in str(op.type):
                    aps.append(op)
                    varlist.append(op.inputs[0])
            
            last_outputs = []
            num_nodes = len(ops)
            name2nodeIdx_map = {}
            for i in range(num_nodes):
                name2nodeIdx_map[ops[i].name] = i
            node_outputs_ = [[] for i in range(num_nodes)]
            for n in range(num_nodes):
                op = ops[n]
                pending_count = len(op.inputs)
                for i in range(pending_count):
                    input_name_id = op.inputs[i].name.split(':')
                    node_outputs_[name2nodeIdx_map[input_name_id[0]]].append(n)
            for n in range(num_nodes):
                if len(node_outputs_[n]) == 0 and ops[n].type != 'NoOp':
                    print('- {0:20s} {1}'.format(ops[n].type, ops[n].name))
                    for m in range(len(ops[n].inputs)):
                        print('<-in-- {0:20s}'.format(ops[n].inputs[m].name))
                        last_outputs.append(ops[n].inputs[m])

            # Init as Ones
            sess.run(init)
            # Get vals before apply_gradients
            for i in range(warmup):
                ret = sess.run(last_outputs + varlist, feed_dict)
                for i in range(0, len(last_outputs)):
                    out_flat = ret[i].flat
                    if (len(out_flat) > 0):
                        max_len = min(10, len(out_flat))
                        print(last_outputs[i].name)
                        print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")
                # Do the apply_gradient
                sess.run(init)
                ret1 = sess.run(varlist + aps , feed_dict)
                print("Updated:")
                for i in range(0, len(varlist)):
                    print(varlist[i].name, ret1[i])
            
            iter_times = []
            for i in range(num_iter):
                start_time = time.time()
                ret = sess.run(last_outputs + varlist, feed_dict)
                ret1 = sess.run(varlist + aps , feed_dict)
                iter_time = (time.time() - start_time) * 1000
                iter_times.append(iter_time)
                print("Iteration time %f ms" % (iter_time))

            print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
                min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

    def import_graph(self, file):
        graph_def = None
        graph = None

        print('Loading graph definition ...', file=sys.stderr)
        try:
            with tf.gfile.GFile(file, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        except:
            raise Exception('Error loading the graph definition')

        print('Importing graph ...', file=sys.stderr)
        try:
            assert graph_def is not None
            with tf.Graph().as_default() as graph:  # type: tf.Graph
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name='',
                    op_dict=None,
                    producer_op_list=None
                )
        except:
            raise Exception('Error importing the graph')

        
        return graph_def, graph

    def graphdef_to_json(self, fin, fout):
        with gfile.FastGFile(fin,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            with gzip.open(fout, 'wt') as fp:
                print("Saving JSON..")
                fp.write(json_format.MessageToJson(graph_def))
                print("Saving JSON done!")

