from time import time


class Timer:
    def __init__(self, unit="s", color=True):
        self.clear()
        self.unit = unit
        self.color = color

    def clear(self):
        self.min = 1e9
        self.max = 0
        self.sum = 0
        self.cnt = 0

    def start(self):
        self.start_time = time()

    def end(self):
        end = time()
        duration = end - self.start_time
        return self.convert_unit(duration)

    def log(self):
        end = time()
        duration = end - self.start_time
        self.min = min(self.min, duration)
        self.max = max(self.max, duration)
        self.sum += duration
        self.cnt += 1

    def report(self, color = None):
        if color is None: color = self.color
        if color:
            print("\033[31m{} iters, min = {:.4f} {}, max = {:.4f} {}, avg = {:.4f} {}\033[m".format(
                self.cnt,
                self.convert_unit(self.min), self.unit,
                self.convert_unit(self.max), self.unit,
                self.convert_unit(self.sum / self.cnt), self.unit
            ))
        else:
            print("{} iters, min = {:.4f} {}, max = {:.4f} {}, avg = {:.4f} {}".format(
                self.cnt,
                self.convert_unit(self.min), self.unit,
                self.convert_unit(self.max), self.unit,
                self.convert_unit(self.sum / self.cnt), self.unit
            ))
        self.clear()

    def convert_unit(self, t):
        if self.unit == "s":
            return t
        elif self.unit == "ms":
            return t * 1000
        elif self.unit == "us":
            return t * 1e6
        elif self.unit == "ns":
            return t * 1e9
        else:
            raise NotImplementedError
