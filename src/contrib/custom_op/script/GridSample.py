#!/bin/python
from __operator__ import OperatorBase, OperatorConfigSameAsInput, OperatorSingleOutputAsOneInput

# Pure antares will not need shape inference


class GridSample(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None):
        super().__init__(input_dict, self.config_infer)
        self.attach_antares_ir(input_dict)

    def attach_antares_ir(self, input_json):
        # if self["mode"] is "bilinear" and self["padding_mode"] is "border":
        N = self["output"]["shape"][0][0]
        C = self["output"]["shape"][0][1]
        HO = self["output"]["shape"][0][2]
        WO = self["output"]["shape"][0][3]
        HI = self["input"]["shape"][0][2] - 1 
        WI = self["input"]["shape"][0][3] - 1
        # align_corners = True
        tmpl = '''
        h_map[N, HO, WO] = (((input1[N, HO, WO, 0] + 1.0) // 2.0) * @HI@).call(`min`,[const(@HI@).cast(`float16`)]).call(`max`, [0.0]) where N in @N@, HO in @HO@, WO in @WO@;
        w_map[N, HO, WO] = (((input1[N, HO, WO, 1] + 1.0) // 2.0) * @WI@).call(`min`,[const(@WI@).cast(`float16`)]).call(`max`, [0.0]) where N in @N@, HO in @HO@, WO in @WO@;
        output0[N, C, HO, WO] = 
              input0[N, C, h_map[N, HO, WO].cast(`int32`), w_map[N, HO, WO].cast(`int32`)] * (1.0 - h_map[N, HO, WO].call(`remainder`)) * (1.0 - w_map[N, HO, WO].call(`remainder`)) 
            + input0[N, C, h_map[N, HO, WO].cast(`int32`), w_map[N, HO, WO].call(`ceil`)] * (1.0 - h_map[N, HO, WO].call(`remainder`)) * (w_map[N, HO, WO].call(`remainder`)) 
            + input0[N, C, h_map[N, HO, WO].call(`ceil`), w_map[N, HO, WO].cast(`int32`)] * (h_map[N, HO, WO].call(`remainder`)) * (1.0 - w_map[N, HO, WO].call(`remainder`)) 
            + input0[N, C, h_map[N, HO, WO].call(`ceil`), w_map[N, HO, WO].call(`ceil`)] * (h_map[N, HO, WO].call(`remainder`)) * (w_map[N, HO, WO].call(`remainder`)) where N in @N@, C in @C@, HO in @HO@, WO in @WO@;
        '''
        self["antares_ir"] = tmpl.replace("@HI@", str(HI)).replace("@WI@", str(WI)).replace("@N@", str(N)).replace("@HO@", str(HO)).replace("@WO@", str(WO)).replace("@C@", str(C))

    def config_infer(self, input_json):
        # input0 = (N, C, Hin, Win)
        # input1 = (N, Hout, Wout, 2)
        # output0 = (N, C, Hout, Wout)

        outputs = {"shape": [[]], "dtype": []}
        outputs['dtype'].append(input_json["input"]["dtype"][0])
        # N
        outputs['shape'][0].append(input_json["input"]["shape"][0][0])
        # C
        outputs['shape'][0].append(input_json["input"]["shape"][0][1])
        # Hout
        outputs['shape'][0].append(input_json["input"]["shape"][1][1])
        # Wout
        outputs['shape'][0].append(input_json["input"]["shape"][1][2])
        return outputs