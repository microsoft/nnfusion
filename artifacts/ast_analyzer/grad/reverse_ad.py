import collections

import astunparse
import gast as ast
import inspect
import six

from . import annotations as anno
from . import ast_utils, cfg, comments, create, funcsigs, grads, naming, non_differentiable, quoting, template, utils
from .naming import Namer
import copy


def get_store_restore_ret_arg(cache_id, orig_target, loop_level):
    if loop_level.depth() == 0:
        store = template.replace(
            'cache = y', cache=cache_id, y=ast_utils.copy_node(orig_target)
        )
        restore = template.replace(
            'y = cache', cache=cache_id, y=ast_utils.copy_node(orig_target)
        )
    elif loop_level.depth() == 1:
        if loop_level.is_fixed:
            store = template.replace(
                'cache[d0] = y', cache=cache_id, y=ast_utils.copy_node(orig_target), d0=loop_level.get_forward(0)
            )
        else:
            store = template.replace(
                'cache.append(y)', cache=cache_id, y=ast_utils.copy_node(orig_target)
            )
        restore = template.replace(
            'y = cache[d0]', cache=cache_id, y=ast_utils.copy_node(orig_target), d0=loop_level.get_backward(0)
        )
    else:
        raise NotImplementedError

    ret = ast.Name(id=cache_id, annotation=None,
                   ctx=ast.Load(), type_comment=None)
    arg = ast.Name(id=cache_id, annotation=None,
                   ctx=ast.Param(), type_comment=None)
    anno.setanno(store, 'store_stmt', True)
    anno.setanno(restore, 'restore_stmt', True)
    anno.setanno(ret, 'cache_ret', True)
    anno.setanno(arg, 'cache_arg', True)
    nodes = (store, restore, ret, arg)
    anno.setanno(store, 'related_nodes', nodes)
    anno.setanno(restore, 'related_nodes', nodes)
    anno.setanno(ret, 'related_nodes', nodes)
    anno.setanno(arg, 'related_nodes', nodes)
    return nodes


def get_ret_arg(orig_target_id):
    ret = ast.Name(id=orig_target_id, annotation=None,
                   ctx=ast.Load(), type_comment=None)
    arg = ast.Name(id=orig_target_id, annotation=None,
                   ctx=ast.Param(), type_comment=None)
    anno.setanno(ret, 'cache_target_ret', True)
    anno.setanno(arg, 'cache_target_arg', True)
    nodes = (ret, arg)
    anno.setanno(ret, 'related_nodes', nodes)
    anno.setanno(arg, 'related_nodes', nodes)
    return nodes


class ReverseAD(object):
    def __init__(self, wrt, device):
        self.required = []
        self.wrt = wrt
        self.attr_grads = {}
        self.loop_level = ast_utils.LoopLevel()
        self.to_declare = {}  # id -> AST Node of torch.empty()
        self.device = device

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        if not hasattr(self, method):
            raise ValueError('Unknown node type: %s' % node.__class__.__name__)
        visitor = getattr(self, method)

        # If this node is a statement, inform all child nodes what the active
        # variables in this statement are
        if anno.hasanno(node, 'active_in'):
            self.active_variables = anno.getanno(node, 'active_in')

        pri, adj = visitor(node)

        # Annotate primal and adjoint statements
        if isinstance(pri, ast.AST):
            anno.setdefaultanno(pri, 'adj', adj)
        else:
            for node in pri:
                anno.setdefaultanno(node, 'adj', adj)
        if isinstance(adj, ast.AST):
            anno.setdefaultanno(adj, 'pri', pri)
        else:
            for node in adj:
                anno.setdefaultanno(node, 'pri', pri)

        return pri, adj

    def visit_FunctionDef(self, node):
        self.namer = Namer.build(node)
        self.forward_cache = {}  # type: Map[str->(gast.Name, gast.Name)]

        for arg in node.args.args:
            assert(isinstance(arg, ast.Name))
            if arg.id != "self":
                self.forward_cache[arg.id] = get_ret_arg(arg.id)

        return_nodes = [n for n in ast.walk(
            node) if isinstance(n, ast.Return)]
        if ((len(return_nodes) > 1) or not isinstance(node.body[-1], ast.Return)):
            raise ValueError('function must have exactly one return statement')
        return_node = ast_utils.copy_node(return_nodes[0])

        body, adjoint_body = self.visit_statements(node.body[:-1])

        # Annotate the first statement of the primal and adjoint as such
        if body:
            body[0] = comments.add_comment(
                body[0], 'Beginning of forward pass')
        if adjoint_body:
            adjoint_body[0] = comments.add_comment(
                adjoint_body[0], 'Beginning of backward pass')

        # Before updating the primal arguments, extract the arguments we want
        # to differentiate with respect to
        dx = ast.Tuple([create.create_grad(node.args.args[i], self.namer)
                        for i in self.wrt], ctx=ast.Load())
        # dx = ast.Tuple([], ctx=ast.Load())
        dx.elts.extend([ast.Name(id=name, ctx=None, annotation=None,
                       type_comment=None) for name in self.attr_grads.values()])
        self.attrs_order = [node.attr for node in self.attr_grads.keys()]

        for _dx in dx.elts:
            _dx.ctx = ast.Load()
        return_dx = ast.Return(value=dx)

        # TODO: args

        # Rename the function to its primal name
        func = anno.getanno(node, 'func')
        node.name = naming.primal_name(func, self.wrt)

        # The new body is the primal body plus the return statement

        y = node.body[-1].value
        if isinstance(y, ast.Tuple):
            raise NotImplementedError
            # y = y.elts[0]
        ret_nodes = [ast.Name(id=y.id, ctx=ast.Load(), annotation=None, type_comment=None)]
        for n in ret_nodes:
            anno.setanno(n, 'origin_ret', True)
        ret_nodes.extend([n[0] for n in self.forward_cache.values()])
        ret_nodes = list(filter(lambda n: not anno.hasanno(n, 'attr_name'), ret_nodes))
        forward_ret = template.replace(
            "return results",
            results=ast.Tuple(
                elts=ret_nodes,
                ctx=ast.Load()
            )
        )

        node.body = body + [forward_ret]

        dy = ast.Name(id=self.namer.grad(y.id), ctx=ast.Param(),
                      annotation=None, type_comment=None)

        adjoint_template = grads.adjoints[ast.FunctionDef]
        adjoint, = template.replace(adjoint_template, namer=self.namer,
                                    adjoint_body=adjoint_body, return_dx=return_dx)
        # NOTE: different from tangent library
        if node.args.args[0].id == 'self':
            adjoint.args.args.append(node.args.args[0]) # self
        adjoint.args.args.extend([dy])
        adjoint.args.args.extend([n[1] for n in self.forward_cache.values() if not anno.hasanno(n[1], 'attr_name')])
        adjoint.args.args.extend([n[1] for n in self.forward_cache.values() if anno.hasanno(n[1], 'attr_name')])
        adjoint.name = naming.adjoint_name(func, self.wrt)

        return node, adjoint

    def visit_statements(self, nodes):
        """Generate the adjoint of a series of statements."""
        primals, adjoints = [], collections.deque()
        for node in nodes:
            primal, adjoint = self.visit(node)
            if not isinstance(primal, list):
                primal = [primal]
            if not isinstance(adjoint, list):
                adjoint = [adjoint]
            # Methods will return `None` if the node is to be removed, so remove them
            primals.extend(filter(None, primal))
            # We reverse the order of the adjoints, but not the statements in
            # the adjoint itself
            adjoints.extendleft(filter(None, adjoint[::-1]))
        return primals, list(adjoints)

    def visit_Assign(self, node):
        """Visit assignment statement."""
        if len(node.targets) != 1:
            raise ValueError('no support for chained assignment')

        # Before the node gets modified, get a source code representation
        # to add as a comment later on
        if anno.hasanno(node, 'pre_anf'):
            orig_src = anno.getanno(node, 'pre_anf')
        else:
            orig_src = quoting.unquote(node)

        # Set target for the RHS visitor to access
        orig_target = ast_utils.copy_node(node.targets[0])

        cache_id = self.namer.cache_fwd(orig_target)

        if isinstance(orig_target, ast.Tuple):
            target_is_tuple = True
            stores = []
            restores = []
            rets = []
            args = []
            cache_ids = []
            for i, target in enumerate(orig_target.elts):
                sub_id = f"{cache_id}_{i}"
                store_, restore_, ret_, arg_ = get_store_restore_ret_arg(sub_id, target, self.loop_level)
                stores.append(store_)
                restores.append(restore_)
                rets.append(ret_)
                args.append(arg_)
                cache_ids.append(sub_id)
        else:
            target_is_tuple = False
            store, restore, ret, arg = get_store_restore_ret_arg(
                cache_id, orig_target, self.loop_level)

        if target_is_tuple:
            resets = []
            for target, cid, ret, arg in zip(node.targets[0].elts, cache_ids, rets, args):
                if self.loop_level.depth() > 0:
                    buf_tensor = self.loop_level.tensor_of_type(
                        anno.getanno(target, 'type'), device=self.device)
                    assert cid not in self.to_declare # !!!!!
                    self.to_declare[cid] = buf_tensor # !!!!!
                assert(cid not in self.forward_cache) # !!!!! ret arg cache_id
                self.forward_cache[cid] = (ret, arg)
                if target.id not in self.forward_cache:
                    ret, arg = get_ret_arg(target.id)
                    self.forward_cache[target.id] = (ret, arg)
                    if anno.hasanno(node, 'attr_name'):
                        attr_name = anno.getanno(node, 'attr_name')
                        anno.setanno(ret, 'attr_name', attr_name)
                        anno.setanno(arg, 'attr_name', attr_name)
                reset = template.replace(
                    'd[y] = zero', y=target, zero=ast_utils.generate_zero_ast(target, anno.getanno(target, 'type'), self.device), namer=self.namer, replace_grad=template.Replace.FULL
                )
                resets.append(reset)
                # print("[reset]", astunparse.unparse(reset))
        else:
            if self.loop_level.depth() > 0:
                buf_tensor = self.loop_level.tensor_of_type(
                    anno.getanno(node.value, 'type'), device=self.device)
                assert cache_id not in self.to_declare # !!!!!
                self.to_declare[cache_id] = buf_tensor # !!!!!
            assert(cache_id not in self.forward_cache) # !!!!! ret arg cache_id
            self.forward_cache[cache_id] = (ret, arg)
            if orig_target.id not in self.forward_cache:
                ret, arg = get_ret_arg(orig_target.id)
                self.forward_cache[orig_target.id] = (ret, arg)
                if anno.hasanno(node, 'attr_name'):
                    attr_name = anno.getanno(node, 'attr_name')
                    anno.setanno(ret, 'attr_name', attr_name)
                    anno.setanno(arg, 'attr_name', attr_name)
            # self.forward_cache.add(self.orig_target.id)
            reset = template.replace(
                'd[y] = zero', y=orig_target, zero=ast_utils.generate_zero_ast(orig_target, anno.getanno(orig_target, 'type'), self.device), namer=self.namer, replace_grad=template.Replace.FULL
            )


        # If there are no active nodes, we don't need to find an adjoint
        # We simply store and restore the state, and reset the gradient
        if not self.is_active(node) and not ast_utils.is_attr_of(node.value, 'self'):
            if target_is_tuple:
                return stores + [node], restores + resets
            else:
                return [store, node], [restore, reset] # !!!!!

        # We create a temporary variable for the target that the RHS can use
        self.target = create.create_temp(orig_target, self.namer)
        create_tmp = template.replace(
            'tmp = y', tmp=ast_utils.copy_node(self.target), y=ast_utils.copy_node(orig_target))

        # Get the primal and adjoint of the RHS expression
        try:
            fx, adjoint_rhs = self.visit(node.value)
        except ValueError as e:
            context = [t.id if hasattr(t, 'id') else t for t in node.targets]
            raise ValueError(
                'Failed to process assignment to: %s. Error: %s' % (context, e))

        if not isinstance(adjoint_rhs, list):
            adjoint_rhs = [adjoint_rhs]

        # Walk the RHS adjoint AST to find temporary adjoint variables to sum
        accumulations = []
        for n in adjoint_rhs:
            for succ in ast.walk(n):
                if anno.hasanno(succ, 'temp_adjoint_var'):
                    xi = anno.getanno(succ, 'temp_adjoint_var')
                    dxi_partial = ast_utils.copy_node(succ)
                    partial = anno.getanno(dxi_partial, 'temp_adjoint_var')
                    accu = template.replace(
                        'd[xi] = d[xi] + grad.unbroadcast(dxi_partial, d[xi])',
                        namer=self.namer, replace_grad=template.Replace.FULL,
                        xi=xi, dxi_partial=dxi_partial)
                    anno.setanno(accu, 'add_grad', True)
                    anno.setanno(accu.value.right, 'type', anno.getanno(xi, 'type'))
                    accumulations.append(accu)

        # The primal consists of storing the state and then performing the
        # assignment with the new primal.
        # The primal `fx` may be optionally (but rarely) redefined when the
        # adjoint is generated, in `fx, adjoint_rhs = self.visit(node.value)`.
        # If we see that the primal value is an Assign node, or a list of nodes
        # (with at least one being an Assign) node, we allow the primal to change.
        # Otherwise, we'll build our own Assign node.
        if isinstance(fx, ast.Assign):
            assign = [fx]
        elif (isinstance(fx, list) and
              any([isinstance(ifx, ast.Assign) for ifx in fx])):
            assign = fx
        else:
            assign = template.replace(
                'y = fx', y=ast_utils.copy_node(orig_target), fx=fx)
            assign = [assign]

        if target_is_tuple:
            primal = stores + assign
            adjoint = [create_tmp] + restores + adjoint_rhs
        else:
            primal = [store] + assign # !!!!!
            adjoint = [create_tmp, restore] + adjoint_rhs # !!!!!

        if target_is_tuple:
            for r in resets:
                if isinstance(node.value, ast.Attribute):
                    self.attr_grads[node.value] = r.targets[0].id
                else:
                    adjoint.append(r)
        else:
            if isinstance(node.value, ast.Attribute):
                self.attr_grads[node.value] = reset.targets[0].id
            else:
                adjoint.append(reset)

        adjoint += accumulations

        if (isinstance(orig_target, ast.Subscript) and
                isinstance(orig_target.slice.value, ast.Name)):
            raise NotImplementedError

        # Add a comment in the backwards pass, indicating which
        # lines in the forward pass generated the adjoint
        for i, adj in enumerate(adjoint):
            adjoint[i] = comments.add_comment(adj, 'Grad of: %s' % orig_src)
        return primal, adjoint

    def is_active(self, node):
        """Checks whether a statement is active.

        An assignment is active when its right hand side contains active
        variables.

        Args:
        node: an instance of ast.Assign

        Returns:
        Whether the statement is active.
        """
        # Special case: If the right hand side is a pop statement, we want to
        # process it
        # if (isinstance(node.value, ast.Call) and
        #     anno.getanno(node.value, 'func', False) == utils.pop):
        #     return True
        for succ in ast.walk(node.value):
            if (isinstance(succ, ast.Name) and isinstance(succ.ctx, ast.Load) and
                    succ.id in self.active_variables):
                return True
        return False

    def visit_Call(self, node):
        """Create adjoint for call.

        We don't allow unpacking of parameters, so we know that each argument
        gets passed in explicitly, allowing us to create partials for each.
        However, templates might perform parameter unpacking (for cases where
        the number of arguments is variable) and express their gradient as a
        tuple. In this case, we have to unpack this tuple of partials.
        """
        # Find the function we are differentiating
        func = anno.getanno(node, 'func')
        if func in non_differentiable.NON_DIFFERENTIABLE:
            return node, []
        if func not in grads.adjoints:
            raise NotImplementedError("backward:", func)
        # We have a template for the gradient that we need to fill in
        template_ = grads.adjoints[func]

        # Match the function call to the template
        sig = funcsigs.signature(template_)
        sig = sig.replace(parameters=list(sig.parameters.values())[1:])
        kwargs = dict((keyword.arg, keyword.value)
                      for keyword in node.keywords)
        bound_args = sig.bind(*node.args, **kwargs)

        # Fill in any missing kwargs with the defaults from the template
        args = quoting.parse_function(template_).body[0].args
        kwargs = dict(zip(*map(reversed, [args.args, args.defaults])))
        kwargs.update(dict(zip(args.kwonlyargs, args.kw_defaults)))
        for arg, val in kwargs.items():
            if arg.id not in bound_args.arguments:
                bound_args.arguments[arg.id] = val

        # Let's fill in the template. The first argument is the output, which
        # was stored in a temporary variable
        output_name = six.get_function_code(template_).co_varnames[0]
        arg_replacements = {output_name: ast_utils.copy_node(self.target)}
        arg_replacements.update(bound_args.arguments)

        # If the template uses *args, then we pack the corresponding inputs
        packing = []
        flags = six.get_function_code(template_).co_flags

        if flags & inspect.CO_VARARGS:
            to_pack = node.args[six.get_function_code(
                template_).co_argcount - 1:]
            vararg_name = six.get_function_code(template_).co_varnames[-1]
            target = ast.Name(annotation=None, id=vararg_name, ctx=ast.Store())
            value = ast.Tuple(elts=to_pack, ctx=ast.Load())
            packing = [ast.Assign(targets=[target], value=value)]

            # And we fill in the packed tuple into the template
            arg_replacements[six.get_function_code(
                template_).co_varnames[-1]] = target

        adjoint = template.replace(
            template_, namer=self.namer, **arg_replacements)
        unpacking = []
        if flags & inspect.CO_VARARGS:
            # If the template packs arguments, then we have to unpack the
            # derivatives afterwards
            # We also have to update the replacements tuple then
            dto_pack = [create.create_temp_grad(arg, self.namer)
                        for arg in to_pack]
            value = create.create_grad(target, self.namer)
            target = ast.Tuple(elts=dto_pack, ctx=ast.Store())
            unpacking = [ast.Assign(targets=[target], value=value)]

        return node, packing + adjoint + unpacking

    def visit_BinOp(self, node):
        op = type(node.op)
        if op not in grads.adjoints:
            raise ValueError('unknown binary operator')
        adjoint_template = grads.adjoints[op]
        adjoint = template.replace(adjoint_template,
                                   namer=self.namer,
                                   x=node.left, y=node.right, z=self.target)
        return node, adjoint

    def visit_Attribute(self, node):
        return node, []


    def visit_Pass(self, node):
        return node, []
    

    def extract_iter(self, node):
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) \
                and node.iter.func.id == 'range' and len(node.iter.args) == 1:
            # for i in range(xxx): ->
            # tmp = xxx; for i in range(tmp):

            cache_id = self.namer.cache_fwd(node.target)
            it = node.iter
            iter_assign = template.replace(
                'cache = y', cache=cache_id, y=node.iter.args[0])
            node.iter.args[0] = ast.Name(id=cache_id, annotation=None,
                                         ctx=ast.Load(), type_comment=None)
            if anno.hasanno(it, 'type'):
                anno.setanno(node.iter.args[0], 'type', copy.copy(
                    anno.getanno(it, 'type')))
            return iter_assign
        else:
            raise NotImplementedError
        # TODO: if not_predictable: return false

    def visit_For(self, node):
        if node.orelse:
            raise ValueError
        iter_assign = self.extract_iter(node)

        counter = self.namer.counter()
        ret, arg = get_ret_arg(counter)
        self.forward_cache[counter] = (ret, arg)
        counter_bwd = self.namer.counter()
        self.loop_level.add_level(
            counter, counter_bwd, ast_utils.copy_node(iter_assign.targets[0]))

        tensor_type = self.loop_level.tensor_of_type(
            anno.getanno(iter_assign.value, 'type'), device=self.device)

        tmp_target = create.create_temp(node.target, self.namer)
        self.to_declare[tmp_target.id] = tensor_type

        save_target, load_target, ret, arg = get_store_restore_ret_arg(
            tmp_target.id, node.target, self.loop_level)
        self.forward_cache[tmp_target] = (ret, arg)

        body, adjoint_body = self.visit_statements(node.body)

        primal = template.replace(
            grads.primals[ast.For],
            body=body,
            i=counter,
            iter_=node.iter,
            target=node.target,
            save_target=save_target
        )

        adjoint = template.replace(
            grads.adjoints[ast.For],
            adjoint_body=adjoint_body,
            n=counter,
            i2=counter_bwd,
            i_tmp=self.namer.counter(),
            load_target=load_target
        )

        self.loop_level.del_level()

        buffers = [iter_assign]
        if self.loop_level.depth() == 0:
            for name, tensor in self.to_declare.items():
                buffers.append(template.replace(
                    "target = tensor", target=name, tensor=tensor.value))

        primal = buffers + primal

        return primal, adjoint

    def visit_Subscript(self, node):
        adjoint = template.replace('d[x] = d[y]', namer=self.namer,
                                   y=ast_utils.copy_node(self.target), x=ast_utils.copy_node(node))
        return node, adjoint


def reverse_ad(node, device):
    """Perform reverse-mode AD on an AST.

    This function analyses the AST to determine which variables are active and
    proceeds by taking the naive derivative. Before returning the primal and
    adjoint it annotates push and pop statements as such.

    Args:
        node: A `FunctionDef` AST node.
        wrt: A tuple of argument indices with respect to which we take the
            derivative.
        preserve_result: A boolean indicating whether the generated
            derivative function should also return the original return value.
        check_dims: A boolean indicating whether the seed derivatives should have
            their dimensions checked to match their primal counterpart.


    Returns:
        mod: A `Module` node containing the naive primal and adjoint of the
            function which can be fed to the `split` and `joint` functions.
        required: A list of tuples of functions and argument indices. These
            functions were called by the function but did not have an adjoint.
    """
    if not isinstance(node, ast.FunctionDef):
        raise TypeError(str(type(node)))
    # Activity analysis
    if node.args.args[0].id == 'self':
        wrt = tuple(range(1, len(node.args.args)))
    else:
        wrt = tuple(range(0, len(node.args.args)))

    cfg.forward(node, cfg.Active(wrt))

    ad = ReverseAD(wrt, device)
    pri, adj = ad.visit(node)
    mod = ast.Module(body=[pri, adj], type_ignores=[])
    return mod, ad.attrs_order
