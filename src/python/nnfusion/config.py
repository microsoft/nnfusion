class Config(dict):
    def __init__(self,
		 *args,
            	 antares_mode=True,
            	 blockfusion_level=0,
            	 extern_result_memory=True,
            	 function_codegen=True,
            	 ir_based_fusion=False,
            	 kernel_fusion_level=0,
            	 kernel_tuning_steps=1000,
		 **kwargs):

        locals_ = locals()
        super().__init__({
            flag: locals_[flag]
            for flag in self.__init__.__kwdefaults__
        })
        super().__init__(*args, **kwargs)

    @staticmethod
    def _parse_flag_value(flag, value):
        value = int(value) if isinstance(value, bool) else value
        return f'-f{flag}={value}'

    def to_flag(self):
        return ' '.join([
            self._parse_flag_value(flag, self[flag])
            for flag in sorted(self.keys())
        ])

