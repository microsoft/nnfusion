from collections.abc import MutableMapping


class Config(MutableMapping):
    """
    NNFusion compilation config. Can pass in any other NNFusion compiler flags
    (execute the command `nnfusion` in the terminal for more details) and
    unknown flags will be ignored.
    Use it as a `dict` with some default key-value pairs.
    """
    def __init__(self,
                 *args,
                 antares_mode=True,
                 blockfusion_level=0,
                 extern_result_memory=True,
                 function_codegen=True,
                 ir_based_fusion=False,
                 kernel_fusion_level=0,
                 kernel_tuning_steps=1000,
                 cuda_set_device=False,
                 **kwargs):
        locals_ = locals()

        self._storage = {
            flag: locals_[flag]
            for flag in self.__init__.__kwdefaults__
        }
        self._storage.update(dict(*args, **kwargs))

    @staticmethod
    def _parse_flag_value(flag, value):
        value = int(value) if isinstance(value, bool) else value
        return f'-f{flag}={value}'

    def to_flag(self):
        return ' '.join([
            self._parse_flag_value(flag, value)
            for flag, value in sorted(self._storage.items())
        ])

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __delitem__(self, key):
        del self._storage[key]
