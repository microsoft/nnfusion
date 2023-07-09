from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import torch

class PipelineManager():
    def __init__(self, num_stages) -> None:
        self.num_stages = num_stages
        self.pipe = [ThreadPoolExecutor(max_workers=1) for _ in range(self.num_stages)]
        self.buffer = [None for _ in range(self.num_stages)]
        self.funcs = [None for _ in range(self.num_stages)]

    def set_funcs(self, funcs) -> None:
        self.funcs = funcs

    def run(self, idx_limit):
        waiting_tasks = {}
        for pipe_idx in range(idx_limit + self.num_stages - 1):
            for i in reversed(range(self.num_stages)):
                run_id = pipe_idx - i
                if run_id < 0 or run_id >= idx_limit:
                    continue
                inputs = None if i == 0 else self.buffer[i - 1]
                waiting_tasks[i] = self.pipe[i].submit(self.funcs[i], args=[run_id, inputs])
            _ = wait(waiting_tasks.values(), return_when=ALL_COMPLETED)
            for i in range(self.num_stages):
                if i in waiting_tasks:
                    self.buffer[i] = waiting_tasks[i].result()
                else:
                    self.buffer[i] = None
            waiting_tasks.clear()

class PipelineGlobal():
    device_to_streams = {}
    pipeline = None
    @classmethod
    def get_streams(cls, device, name):
        if device not in cls.device_to_streams:
            cls.device_to_streams[device] = {"h2d" : torch.cuda.Stream(device), "d2h" : torch.cuda.Stream(device), "kernel" : torch.cuda.Stream(device)}
        return cls.device_to_streams[device][name]

    @classmethod
    def get_pipeline(cls):
        if cls.pipeline is None:
            cls.pipeline = PipelineManager(5)
        return cls.pipeline

def stage1(args):
    idx, x = args
    return idx

def stage2(args):
    idx, x = args
    return x + 1

def stage3(args):
    idx, x = args
    print(x)
    return None

if __name__ == "__main__":
    p = PipelineManager(3)
    p.set_funcs([stage1, stage2, stage3])
    p.run(5)
