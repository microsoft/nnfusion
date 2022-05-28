from arch.Arch import Arch
from lang.generic import output
from memopt.graph import find_topo_sort
import numpy as np
from memopt.bestfit import BestFit
from .utils import TileInfo
from queue import PriorityQueue
import math

def get_all_factors(n: int):
    n0 = int(np.ceil(np.sqrt(n)))
    val = np.where(n % np.arange(1, n0) == 0)[0] + 1
    mid = np.array([], dtype=int) if n0 * n0 != n else [n0]
    return list(np.concatenate([val, mid, n // val[::-1]]))

def factorize(n: int):
    i = 2
    result = []
    while n > 1:
        if n % i == 0:
            n //= i
            result.append(i)
        else:
            i += 1
    return result

def coalesced_factor(subtensor, tensor):
    if subtensor[-1] != tensor[-1] or len(subtensor) == 1:
        return subtensor[-1]
    else:
        return subtensor[-1] * coalesced_factor(subtensor[:-1], tensor[:-1])

def coalesced_tensor_shape(subtensor, tensor, transaction_size):
    bytes = np.prod(subtensor)
    factor = coalesced_factor(subtensor, tensor)
    return transaction_size * bytes / min(transaction_size, factor)

def get_block_size(n):
    if n < 64:
        return n
    elif n % 32 == 0:
        for blk in [128, 96, 160, 64, 192, 224, 32]:
            if n % blk == 0:
                return blk
    else:
        factors = get_all_factors(n)
        factors = list(filter(lambda x: x <= 1024, factors))
        roundup = [(n + 31) // 32 * 32 for n in factors]
        pad = [(roundup[i] - factors[i]) / factors[i] for i in range(len(factors))]
        score = [abs(factors[i] - 128) * pad[i] for i in range(len(factors))]
        return min(zip(score, factors))[1]

class DefaultPolicy:
    def __init__(self, output_nodes, arch:Arch) -> None:
        self.arch = arch
        self.ordered_nodes = list(filter(
            lambda n: not n.is_placeholder() and not n.is_output(),
            find_topo_sort(output_nodes)
        ))
        self.output_nodes = output_nodes

        # print("subgraph has {} output".format(len(self.output_nodes)))


    def emit_config(self, topk):
        base_tile = self.get_base_tile()
        rstep_map = {node : self._assign_reduce_step(node) for node in self.ordered_nodes}
        # print(rstep_map)
        smem_tile_condidates = self.DFS_smem_tile(base_tile, topk, rstep_map)[:topk]
        results = []
        for tile, info in smem_tile_condidates:
            final_rstep_map = self._expand_reduce_axis(tile, info, rstep_map)
            # print(tile, final_rstep_map, info.smem_cost, info.num_wave, info.block_per_SM)
            codegen_dicts = self.assign_block_size(tile)
            for node in self.ordered_nodes:
                for ax in node.raxis:
                    codegen_dicts[node][ax] = [final_rstep_map[node][ax], 1]
            # print(codegen_dicts)
            results.append(codegen_dicts)
        return results

    def DFS_smem_tile(self, init_tile, topk, rstep_map):
        _steps = [get_all_factors(n) for n in self.output_nodes[0].get_shape()]
        steps = [step[step.index(t):] for step, t in zip(_steps, init_tile)]
        for i in range(len(steps)):
            added = list(filter(lambda s:s < steps[i][-1] and s > steps[i][0] and s not in steps[i], [2, 4, 8, 16, 32]))
            steps[i].extend(added)
            steps[i] = sorted(steps[i])
        visited_tile = {}
        queue = PriorityQueue()
        def prio(info):
            return (info.footprint + 1) * info.num_wave # * (info.block_per_SM ** 0.5)
        def add_to_queue(tile):
            if tuple(tile) in visited_tile:
                return
            tile_map = self.get_tile_map(tile)
            info = self.compute_smem_tile_meta_data(tile_map, rstep_map)
            visited_tile[tuple(tile)] = info
            if info.valid():
                queue.put([prio(info), tile])

        add_to_queue(init_tile)
        while not (queue.empty()):
            _, tile = queue.get()
            dim_ids = [step.index(t) for step, t in zip(steps, tile)]
            for i in reversed(range(len(dim_ids))):
                if dim_ids[i] + 1 < len(steps[i]):
                    new_tile = tile.copy()
                    new_tile[i] = steps[i][dim_ids[i] + 1]
                    add_to_queue(new_tile)

        visited_tile = {k : v for k, v in visited_tile.items() if v.valid()}
        sorted_tiles = sorted(visited_tile.items(), key=lambda x:prio(x[1]))
        return sorted_tiles

    # get the minimum tile that could satisfy no redundancy computation
    def get_base_tile(self):
        out_node = self.output_nodes[0]
        shape = out_node.get_shape()
        base_tile = [1 for _ in shape]
        wpi = self.compute_workload_per_item(base_tile)
        for dim, n in enumerate(shape):
            # factors = get_all_factors(n)
            factors = [n]
            for factor in factors:
                if factor == base_tile[dim]:continue
                tile = base_tile.copy()
                tile[dim] = factor
                new_wpi = self.compute_workload_per_item(tile)
                if new_wpi < wpi:
                    wpi, base_tile = new_wpi, tile
                else:
                    break

        self.op_tile_map = self.get_tile_map(tile)

        if self._assign_basic_tile(tile):
            print("Fail to find base tile")
            return -1

        return base_tile

    # get_tile_map handles multiple output cases
    def get_tile_map(self, tile):
        tile_map = {}
        for node in self.output_nodes:
            tile_map[node] = [
                tile[i] * node.get_shape()[i] // self.output_nodes[0].get_shape()[i] for i in range(len(tile))]
        return tile_map

    def compute_workload_per_item(self, output_tile):
        output_node = self.output_nodes[0]
        queue = [(output_node, output_tile)]
        compute = 0
        while len(queue) > 0:
            node, tile = queue.pop(0)
            dep = node.infer_dependency(tile)
            for i, edge in enumerate(node.inputs):
                if not edge.src_node.is_placeholder():
                    subtensor_shape = dep[i]
                    compute += np.prod(subtensor_shape)
                    queue.append((edge.src_node, subtensor_shape))
        return compute / np.prod(output_tile)

    def _assign_basic_tile(self, output_tile):
        out_shape = self.output_nodes[0].get_shape()
        queue = list(self.op_tile_map.items())
        while len(queue) > 0:
            node, tile = queue.pop(0)
            dep = node.infer_dependency(tile)
            for i, edge in enumerate(node.inputs):
                if not edge.src_node.is_placeholder():
                    subtensor_shape = dep[i]
                    shape = edge.src_node.get_shape()
                    if np.prod(subtensor_shape) / np.prod(shape) != np.prod(output_tile) / np.prod(out_shape):
                        print(subtensor_shape, shape, output_tile, out_shape)
                        return True
                    if edge.src_node in self.op_tile_map:
                        assert self.op_tile_map[edge.src_node] == subtensor_shape
                    else:
                        self.op_tile_map[edge.src_node] = subtensor_shape
                        queue.append((edge.src_node, subtensor_shape))
        return False

    def _assign_reduce_step(self, node):
        if len(node.raxis) == 0:
            return []
        raxis = node.raxis
        tile = node.get_shape()
        cur_rstep = {ax : 1 for ax in raxis}
        def _score(rstep):
            score = 0
            shape = node.infer_dependency(tile, rstep=rstep)
            num_steps = np.prod([node.raxis[ax] / rstep[ax] for ax in raxis])
            for edge in node.inputs:
                if edge.src_node.is_placeholder() and "input{}".format(edge.dst_id) in node.reduction_inputs:
                    score -= num_steps * coalesced_tensor_shape(shape[edge.dst_id], edge.src_node.get_shape(), 32)
            return score

        def _enlarge(rstep):
            candidates = []
            for ax in rstep:
                r = rstep.copy()
                r[ax] = min(r[ax] * 2, raxis[ax])
                candidates.append((r, _score(r)))
            best = max(candidates, key=lambda x:x[1])
            return best

        # enlarge rstep to ensure read is coaleased
        cur_score = _score(cur_rstep)
        while True:
            if cur_score == 0:break
            new_rstep, new_score = _enlarge(cur_rstep)
            if new_score <= cur_score:
                break
            else:
                cur_rstep, cur_score = new_rstep, new_score
        return cur_rstep

    def _expand_reduce_axis(self, output_tile, info, rstep_map):
        tile_map = self.get_tile_map(output_tile)
        _, tile_map = self._compute_memory_footprint(tile_map)
        cur_block_per_SM = info.block_per_SM
        smem_limit = min(self.arch.max_smem_usage // cur_block_per_SM, self.arch.mem_cap(0))
        def _optimize(node, rstep):
            def _score(rstep):
                score = 0
                shape = node.infer_dependency(tile_map[node], rstep=rstep)
                for edge in node.inputs:
                    if edge.src_node.is_placeholder():
                        factor = coalesced_factor(shape[edge.dst_id], edge.src_node.get_shape())
                        score += factor
                return score
            def _enlarge(rstep):
                candidates = []
                for ax in rstep:
                    r = rstep.copy()
                    r[ax] = min(r[ax] * 2, node.raxis[ax])
                    candidates.append((r, _score(r)))
                best = max(candidates, key=lambda x:x[1])
                return best

            cur_score = _score(rstep)
            new_rstep_map = rstep_map.copy()
            while True:
                if cur_score == 0:break
                new_rstep, new_score = _enlarge(rstep)
                new_rstep_map[node] = new_rstep
                if new_score <= cur_score or self._compute_shared_memory_usage(tile_map, new_rstep_map) > smem_limit:
                    break
                else:
                    rstep, cur_score = new_rstep, new_score
            return rstep
        rstep_map = rstep_map.copy()
        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map[node])
                rstep_map[node] = rstep
        return rstep_map

    def _compute_memory_footprint(self, tile_map):
        tile_map = tile_map.copy()
        queue = [node for node in self.output_nodes]
        footprint = 0
        while len(queue) > 0:
            node = queue.pop(0)
            dep = node.infer_dependency(tile_map[node])
            for i, edge in enumerate(node.inputs):
                if edge.src_node.is_placeholder():
                    footprint += np.prod(dep[i])
                elif edge.src_node not in tile_map:
                    tile_map[edge.src_node] = dep[i]
                    queue.append(edge.src_node)
        for node in self.output_nodes:
            footprint += coalesced_tensor_shape(tile_map[node], node.get_shape(), 8)
        # output_shape = np.prod(tile_map[self.output_nodes[0]])
        # footprint /= output_shape
        return footprint, tile_map

    def _compute_shared_memory_usage(self, tile_map, rstep_map):
        allocator = BestFit()
        block_map = {}
        processed = set()
        def can_free(node, out_id):
            for edge in node.outputs:
                if edge.src_id == out_id and edge.dst_node not in processed:
                    return False
            return True
        for node in self.ordered_nodes:
            node_internal_bytes = node.infer_smem_usage(tile_map[node], rstep_map[node])
            block = allocator.malloc(node_internal_bytes)
            allocator.free(block)
            # free inputs
            processed.add(node)
            for edge in node.inputs:
                if not edge.src_node.is_placeholder() and can_free(edge.src_node, edge.src_id):
                    allocator.free(block_map.pop((edge.src_node, edge.src_id)))
            # alloc outputs
            for edge in node.outputs:
                if not edge.dst_node.is_output() and (node, edge.src_id) not in block_map:
                    block_map[(node, edge.src_id)] = allocator.malloc(np.prod(tile_map[node]) * 4)

        assert len(block_map) == 0
        return allocator.limit


    def compute_smem_tile_meta_data(self, output_tile_map, rstep_map) -> TileInfo:
        footprint, tile_map = self._compute_memory_footprint(output_tile_map)
        smem_cost = self._compute_shared_memory_usage(tile_map, rstep_map)
        if smem_cost > self.arch.mem_cap(0):
            return TileInfo(-1, -1, -1, -1)
        out_node = self.output_nodes[0]
        out_tile = output_tile_map[out_node]
        out_shape = out_node.get_shape()
        grid_size = np.prod([np.ceil(y / x) for x, y in zip(out_tile, out_shape)])
        reg_usage = 2 * max([np.prod(tile_map[node]) for node in self.ordered_nodes]) # estimated reg usage
        if reg_usage > self.arch.reg_cap[0]:
            return TileInfo(-1, -1, -1, -1)
        block_per_SM = min(
            self.arch.max_smem_usage // max(smem_cost, 1),
            self.arch.reg_cap[0] // reg_usage,
            4)
        num_wave = np.ceil(grid_size / (block_per_SM * self.arch.compute_max_core[0])) # self.arch.compute_max_core[0]
        return TileInfo(footprint, smem_cost, block_per_SM, num_wave)

    def assign_block_size(self, output_tile):
        tile_map = self.get_tile_map(output_tile)
        _, tile_map = self._compute_memory_footprint(tile_map)
        max_block_size = np.prod(output_tile)
        for node in self.ordered_nodes:
            max_block_size = math.gcd(max_block_size, np.prod(tile_map[node]))
        result = {}
        recommended_block_size = get_block_size(max_block_size)
        for node in self.ordered_nodes:
            result[node] = self._assign_block_size(node, tile_map[node], recommended_block_size)
        return result

    def _assign_block_size(self, node, tile, block_size):
        factors = factorize(block_size)
        cur_threads = [1 for _ in tile]
        rstep = {ax : 1 for ax in node.raxis}
        ndim = len(tile)

        def _score(node, thread): # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.infer_dependency(block_tile)
            for edge in node.inputs:
                score += np.prod(shape[edge.dst_id]) / self.arch.memory_bw(1)
            for edge in node.outputs:
                if edge.dst_node in self.output_nodes: # write to global
                    score += coalesced_tensor_shape(thread, node.get_shape(), 8) / self.arch.memory_bw(0)
            return score
        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if cur_threads[i] >= tile[i]:
                    continue
                divisible = (tile[i] % (cur_threads[i] * factor)) != 0
                cur_threads[i] *= factor
                score_map[i] = (divisible, _score(node, cur_threads), i)
                cur_threads[i] //= factor
            assert len(score_map) > 0
            dim_order = sorted(score_map.keys(), key=lambda x:score_map[x])
            cur_threads[dim_order[0]] *= factor
        # print(cur_threads)
        block_tile = [int(np.ceil(tile[i] / cur_threads[i])) for i in range(ndim)]
        codegen_dict = {ax : [cur_threads[i], block_tile[i]] for i, ax in enumerate(node.saxis)}
        return codegen_dict
