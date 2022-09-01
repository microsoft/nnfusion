from roller.config import *
from roller.cost_model import *
from .policy_base import *
import math
import numpy
from roller.utils import *
from copy import deepcopy

log_regular_tile_found_key = 'regular_tile_found'


def size_of(dtype):
  if dtype == 'float':
    return 4
  if dtype == 'half':
    return 2


def lcm(x, y):
  return x * y // math.gcd(x, y)


def num_tiles(large_tile, base_tile):
  ret = 1
  for d1, d2 in zip(large_tile, base_tile):
    ret *= math.ceil(d1 / d2)
  return ret


def eligible(op, arch, rprog, mem_level, tile_tensor='output'):
  # a tile size is not eligible if
  # 1, the # of used registers exceed the capacity
  # 2, the size of memory used at certain level exceeds the capacity
  if mem_level == 1:  # reg tiling
    # check register usage
    regTile = rprog.GetTile(mem_level)
    reg_usage_2 = op.RegUsage(regTile, tile_tensor)
    if reg_usage_2 > arch._reg_cap(1):
      return False
  if mem_level == 0:  # smem tiling
    num_threads = rprog.GetParallelism(mem_level + 1)
    regTile = rprog.GetTile(mem_level + 1)
    # check register usage
    reg_usage_2 = op.RegUsage(regTile, tile_tensor)
    reg_usage_1 = reg_usage_2 * num_threads
    if reg_usage_1 > arch._reg_cap(0):
      return False
    # check shared memory usage
    smemTile = rprog.GetTile(mem_level)
    if op.MemFootprint(smemTile, tile_tensor) >= arch.mem_cap(0):
      return False
    # remove over sized warps
    if op.use_tc:
      num_threads *= 32
    return num_threads <= 512
  return True


def divisible(large_tile, base_tile):
  for d1, d2 in zip(large_tile, base_tile):
    if d1 % d2 > 0:
      return False
  return True


def Prod(tile):
  ret = 1
  for d in tile:
    ret *= d
  return ret


def DataReuseScore(op, rtile, tile_tensor='output'):
  """
    return a list of scores on each dimension
    """
  dims = rtile.Dimensions().copy()
  ret_list = []
  for d in range(len(dims)):
    new_dims = dims.copy()
    new_dims[d] += 1
    new_rtile = rTile(rtile.expr, new_dims, op.SAxis(), op.RAxis(),
                      op.GetTvmOutTensor())
    compute_workload = op.ComputeWorkload(new_rtile)
    memory_tensors = op.MemWorkload(new_rtile)
    memory_workload = sum(memory_tensors[0]) + sum(memory_tensors[1])
    # negative for decreasing order after sorting
    ret_list.append(-compute_workload / memory_workload)
  return ret_list


def Estimate_ActiveBlock(op, arch, rprog):
  smem_tile = rprog.GetTile(0)
  reg_tile = rprog.GetTile(1)
  smem_usage = op.MemFootprint(smem_tile)
  block_size = rprog.GetParallelism(1)
  return min(
      int(arch.max_smem_usage // smem_usage),
      int(arch.max_threads_per_sm // block_size), arch.max_active_blocks)


def RewriteSche_BankSize(schedule, smem_bank_size):
  '''
    bank size change here
    TODO: data type = float, 4B by default
    '''
  level2_tile, level2_redu = schedule.get_tile(2)
  level1_tile, level1_redu = schedule.get_tile(1)
  merge_number = smem_bank_size // 4
  # inner_axis_id = len(level2_tile) - 1

  if level2_tile[-1] < merge_number:
    expand = merge_number // level2_tile[-1]
    new_level2_tile = list(level2_tile[:-1]) + [merge_number]
    schedule.update_tile(
        mem_level=2, dim=new_level2_tile, reduction_dict=level2_redu)

    if level1_tile[-1] >= expand:
      new_level1_tile = list(level1_tile[:-1]) + [level1_tile[-1] // expand]
      schedule.update_tile(
          mem_level=1, dim=new_level1_tile, reduction_dict=level1_redu)
    else:
      # TODO redundant computation in this branch, report or other solution?
      new_level1_tile = list(level1_tile[:-1]) + [1]
      schedule.update_tile(
          mem_level=1, dim=new_level1_tile, reduction_dict=level1_redu)
  return schedule


class PolicyRT(PolicyBase):
  """
        Constructing tiling schedules using DFS, hardcode reduction step size for now
    """

  def __init__(self,
               op,
               arch,
               smem_tiling=False,
               reg_tiling=False,
               st_align=False,
               padding_threshold_cap=1.0,
               shrink_tiny=True,
               tile_tensor='output'):
    self.op = op
    self.arch = arch
    self.tile_tensor = tile_tensor
    self.num_level = arch.num_level
    if '_unpad' in self.op.expr.output(0).name:
      outputs = [self.op.expr.input_tensors[0], self.op.expr.output(0)]
    else:
      outputs = [self.op.expr.output(0)]
    self.th_cap = padding_threshold_cap
    self.shrink_tiny = shrink_tiny

    self.saxis, self.raxis = get_axis_names(outputs[0])
    self.tile_dim = len(self.saxis)
    self.step_dim = len(self.raxis)

    self.raw_rprogs = []
    self.all_results = []
    self.in_results = set()

    self.ConstructionLog = {}
    self.ConstructionLog[log_regular_tile_found_key] = False
    self.border_rprogs = [[] for _ in range(self.num_level)]
    self.on_border = set()
    # for storage align
    self.smem_tiling = smem_tiling
    self.reg_tiling = reg_tiling
    self.st_align = st_align

  def DataReuseScore(self, rprog, mem_level, tile_tensor='output'):
    """
        return a list of scores on each dimension
        """
    rtile = rprog.GetTile(mem_level)
    dims = rtile.Dimensions().copy()
    ret_list = []
    for d in range(len(dims)):
      new_dims = dims.copy()
      new_dims[d] += 1
      new_rtile = rTile(rtile.expr, new_dims, self.op.SAxis(), self.op.RAxis(),
                        self.op.GetTvmOutTensor())
      rprog.UpdateTile(new_rtile, mem_level)
      self.update_rtile_storage_padding(
          rprog,
          self.arch,
          mem_level,
          smem_tiling=self.smem_tiling,
          reg_tiling=self.reg_tiling,
          st_align=self.st_align)
      compute_workload = self.op.ComputeWorkload(new_rtile)
      memory_tensors = self.op.MemWorkload(new_rtile)
      memory_workload = sum(memory_tensors[0]) + sum(memory_tensors[1])
      # negative for decreasing order after sorting
      ret_list.append(-compute_workload / memory_workload)
    rprog.UpdateTile(rtile, mem_level)
    return ret_list

  def GetAlignedSteps(self, base_rtile, mem_level):
    """
        returns
            steps: list of possible step choices for each dimension
            new_base_rtile: the basic tile size, inferred based on both arithmetic and architecture info
        """
    # hardcode for two-level tiling now
    full_dim = self.op.Dimensions()
    full_sdim = self.op.SDimensions()
    dim = len(full_dim)
    sdim = len(full_sdim)
    steps_arch = []
    base_sdims = base_rtile.SDimensions()
    base_rdims = base_rtile.RDimensions()

    if mem_level == 1:
      # reg level, using power of 2 tiles
      for d in range(sdim):
        steps_arch.append([x * base_sdims[d] for x in [1, 2, 4, 8, 16, 32]])
      for rd in base_rdims:
        steps_arch.append([rd])

    new_base_sdims = list(base_sdims)
    new_base_rdims = list(base_rdims)
    if mem_level == 0:
      # smem level
      # hardcode: only last level aligned to transaction size
      new_base_sdims[-1] = lcm(
          new_base_sdims[-1],
          self.arch.transaction_size[0] // self.op.InputTypeSize())
      for d in range(sdim):
        steps_arch.append([new_base_sdims[d] * (i + 1) for i in range(32)])

      # hardcode: each reduction axis aligned to axis length or transaction size
      if len(self.raxis) > 0:
        for base_rdim, raxis_name in zip(new_base_rdims, self.raxis):
          base_transaction_size = self.arch.transaction_size[
              0] // self.op.InputTypeSize()
          rlen_cap = min(self.op.GetAxisLen(raxis_name), 32)
          if self.op.use_tc:
            base_transaction_size = 64
            rlen_cap = 64
          new_base_rdim = lcm(base_rdim, base_transaction_size)
          steps_arch.append([])
          while True:
            steps_arch[-1].append(min(new_base_rdim, rlen_cap))
            if new_base_rdim >= rlen_cap:
              break
            new_base_rdim *= 4

    # scan through all steps, remove the ones with too much padding
    steps = []
    for d in range(sdim):
      steps.append([])
      for s in steps_arch[d]:
        padded_dim = math.ceil(full_dim[d] / s) * s
        if padded_dim <= full_dim[d] * (1 + self.padding_threshold):
          steps[-1].append(s)
    # print(steps_arch, dim, sdim, mem_level, self.raxis)
    for d in range(sdim, dim):
      steps.append(steps_arch[d])
    for step in steps:
      if len(step) == 0:
        return steps, None
    new_base_dim = [steps[d][0] for d in range(len(steps))]
    new_base_rtile = rTile(self.op.expr, new_base_dim, self.op.SAxis(),
                           self.op.RAxis(), self.op.GetTvmOutTensor())
    return steps, new_base_rtile

  def IsComputeIntensive(self, rprog, mem_level):
    """
        return True if the schedule is compute intensive
        """
    thisTile = rprog.GetTile(mem_level)

    compute_workload = self.op.ComputeWorkload(thisTile)
    #compute_throughput = self.compute_db.lookup(64,8,8) * self.arch.compute_max_core[0]
    compute_throughput = self.arch.peak_flops  # / self.arch.compute_max_core[0]
    if self.op.use_tc:
      compute_throughput = self.arch.peak_tc_flops
    compute_latency = compute_workload / compute_throughput

    memory_tensors = self.op.MemWorkload(thisTile)
    memory_workload = sum(memory_tensors[0]) + sum(memory_tensors[1])

    memory_throughput = self.arch.memory_bw(
        mem_level)  # / self.arch.mem_max_core[0]
    memory_latency = memory_workload / memory_throughput
    return compute_latency > memory_latency

  def IsPeakComputeTile(self, rprog, mem_level):
    if mem_level == 1:
      return True
      #reg_size = Prod(reg_tile)
      #return reg_size >= 32
    if mem_level == 0:
      #smem_tile = rprog.GetTile(0)
      num_threads = rprog.GetParallelism(1)
      if self.op.use_tc:
        num_threads *= 32
      return num_threads % (self.arch.warp_size *
                            self.arch.compute_sm_partition[1]) == 0
    # scale out
    if mem_level == -1:
      num_blocks = rprog.GetParallelism(0)
      return num_blocks >= 2 * self.arch.compute_sm_partition[0]

  def EnlargeTile(self, last_rprog, rprog, steps, mem_level):
    key = rprog.Dump()
    if key in self.visited:
      return
    self.visited.add(key)
    if len(self.top_results) == self.TOPK:
      return

    def one_level_down(rprog, this_level):
      # print("going down {}".format(schedule.dump_to_string()))
      base_rtile = rprog.GetTile(this_level)
      steps, base_rtile = self.GetAlignedSteps(base_rtile, this_level - 1)
      # valid = True
      # for step in steps:
      #     if len(step) == 0:
      #         valid = False
      # if valid:
      if base_rtile:
        rprog.AddTile(mem_level - 1, base_rtile)
        self.EnlargeTile(rprog, rprog, steps, mem_level - 1)
        rprog.DeleteTile(mem_level - 1)

    # exit if current schedule exceeds memory capacity
    self.update_rtile_storage_padding(rprog, self.arch, mem_level,
                                      self.smem_tiling, self.reg_tiling,
                                      self.st_align)
    if not eligible(self.op, self.arch, rprog, mem_level):
      #if mem_level == 0:
      #    print(last_schedule.dump_to_string(), last_schedule.subtile_count(0, 1))
      if self.IsPeakComputeTile(last_rprog, mem_level):
        last_rprog_key = last_rprog.Dump()
        if last_rprog_key not in self.on_border:
          self.border_rprogs[mem_level].append(last_rprog)
          self.on_border.add(last_rprog_key)
        #print("border case")
        #print(last_schedule.dump_to_string())
      if mem_level > 0 and not self.ConstructionLog[log_regular_tile_found_key]:
        one_level_down(last_rprog, mem_level)
      return

    # check if current schedule is compute saturated and is compute intensive
    # scale-up after scale-out to favor large tiles
    if self.IsPeakComputeTile(rprog, mem_level) and self.IsComputeIntensive(
        rprog, mem_level):
      if mem_level == 0:
        self.ConstructionLog[log_regular_tile_found_key] = True
        self.top_results.append(rprog)
        if len(self.top_results) == self.TOPK:
          return
      else:
        # going one level down
        one_level_down(rprog, mem_level)

    # expand the current level tiles
    # caluate data reuse scores
    r_scores = []
    rtile = rprog.GetTile(mem_level)
    shape = rtile.Dimensions()
    # r_scores = DataReuseScore(self.op, rtile)
    r_scores = self.DataReuseScore(rprog, mem_level)
    x = numpy.array(r_scores)
    dim_order = numpy.argsort(x)
    # enumerate from dimensions with highest scores
    for d in dim_order:
      if len(steps[d]) <= 1:
        continue
      new_rprog = rprog.copy()
      new_shape = shape.copy()
      old_step = steps[d].pop(0)
      new_shape[d] = steps[d][0]
      new_rtile = rTile(rprog.expr, new_shape, self.op.SAxis(), self.op.RAxis(),
                        self.op.GetTvmOutTensor())
      new_rprog.AddTile(mem_level, new_rtile)

      self.EnlargeTile(rprog, new_rprog, steps, mem_level)
      steps[d].insert(0, old_step)

  def expand_reduce_axis(self, config, reduction_axis_name, mem_level):
    #rstep = config._reduction_size[mem_level][reduction_axis_name]
    #config._reduction_size[mem_level][reduction_axis_name] = 32
    base_step = config._reduction_size[mem_level][reduction_axis_name]
    axis_len = self.op.reduction_axis_len()
    if self.activeblock_db.lookup_schedule(config) != None:
      active_blocks = self.activeblock_db.lookup_schedule(config)
      while True:
        new_active_blocks = self.activeblock_db.lookup_schedule(config)
        if new_active_blocks != None:
          last_r = config._reduction_size[mem_level][reduction_axis_name]
        config._reduction_size[mem_level][reduction_axis_name] = min(
            config._reduction_size[mem_level][reduction_axis_name] + base_step,
            axis_len)
        if not eligible(
            self.op, self.arch, config, mem_level
        ) or config._reduction_size[mem_level][reduction_axis_name] > 32:
          config._reduction_size[mem_level][reduction_axis_name] = last_r
          break
        if new_active_blocks != None and new_active_blocks < active_blocks:
          config._reduction_size[mem_level][reduction_axis_name] = last_r
          break
      return config
    else:
      limit = min(32, self.op.reduction_axis_len())
      last_r = config._reduction_size[mem_level][reduction_axis_name]
      #last_r = base_r
      while config._reduction_size[mem_level][reduction_axis_name] < limit:
        last_r = config._reduction_size[mem_level][reduction_axis_name]
        config._reduction_size[mem_level][reduction_axis_name] = min(
            last_r * 2, limit)
        if not eligible(self.op, self.arch, config, mem_level):
          config._reduction_size[mem_level][reduction_axis_name] = last_r
          break
    return config

  def emit_raw_configs(self, padding_threshold=0):
    """
            using BFS to iteratively search for all eligible candidates level by level
        """
    # initialize uni tiles
    mem_level = self.num_level - 1
    uniTile = rTile(self.op.expr, self.op.GetUniSchedule(), self.op.SAxis(),
                    self.op.RAxis(), self.op.GetTvmOutTensor())
    uniProg = rProg(self.arch.num_level, self.op)
    uniProg.AddTile(self.num_level, uniTile)
    uniProg.AddTile(self.num_level - 1, uniTile)

    # initialize key hyperparameters
    self.padding_threshold = padding_threshold
    steps, _ = self.GetAlignedSteps(uniTile, 1)
    self.top_results = []
    self.border_rprogs = [[] for _ in range(self.num_level)]
    self.on_border = set()

    self.visited = set()
    self.EnlargeTile(uniProg, uniProg, steps, mem_level)

    #for schedule in self.top_results:
    #    schedule = self.expand_reduce_axis(schedule, "k", 0)

    return self.top_results

  def try_shrink(self, rprog):
    sdim = len(rprog.saxis)
    while True:
      smem_tile_dim = rprog.GetTile(0).Dimensions()
      # validate grid size, return if grid size >= # of CUs
      grid_size = rprog.GetParallelism(0)
      if grid_size >= self.arch.compute_max_core[0]:
        return rprog

      # otherwise, shrink dimentions based on data reuse
      #print("try shrink small config: {}".format(schedule.dump_to_string()))
      # r_scores = DataReuseScore(self.op, rprog, 0)
      r_scores = self.DataReuseScore(rprog, 0)[:sdim]
      reversed_score_np = numpy.array([-s for s in r_scores])
      dim_order = numpy.argsort(reversed_score_np)
      for d in dim_order:
        if smem_tile_dim[d] == 1:
          continue
        for l in range(self.arch.num_level):
          tile = rprog.GetTile(l)
          tile_sdim = tile.SDimensions()
          tile_sdim[d] = math.ceil(tile_sdim[d] / 2)
          tile_rdim = tile.RDimensions()
          new_tile = rTile(tile.expr, tile_sdim + tile_rdim, self.op.SAxis(),
                           self.op.RAxis(), self.op.GetTvmOutTensor())
          rprog.UpdateTile(new_tile, l)
      # print("config after shrinking: {}".format(rprog.Dump()))

  def emit_config_without_trails(self, topk):
    # directly compute the theoretical performance for each raw configs and pick the optimal k configs
    # will call self.emit_raw_configs()
    self.TOPK = topk
    th = 0

    while th <= self.th_cap and len(self.all_results) < topk:
      self.emit_raw_configs(th)
      # take border cases if no IO intensity satisfied configs
      if len(self.top_results) == 0:
        self.top_results = self.border_rprogs[0][:self.TOPK]
      if len(self.top_results) == 0:
        print('failed to find results with padding threshold {:.1f}'.format(th))
      else:
        print('found {} results with threshold {:.1f}'.format(
            len(self.top_results), th))
        # add current results to all
        for result in self.top_results:
          key = result.Dump()
          if key not in self.in_results:
            self.in_results.add(key)
            self.all_results.append(result)
      th += 0.2

    # handling small configs
    if self.shrink_tiny:
      for rprog in self.all_results:
        rprog = self.try_shrink(rprog)
    return self.all_results[:self.TOPK]

    output_results = []
    for rprog in self.all_results[:self.TOPK]:
      print('init rprog:', rprog.Dump())
      new_sche = RewriteSche_BankSize(rprog, self.arch.smem_bank_size)
      print('updated rprog:', rprog.Dump())
      output_results.append(new_sche)
    return output_results
