# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from .description import IODescription
from .session import PTSession, tensor2desc


def extract_desc_and_device(name, tensor):
    return tensor2desc(tensor, name=name), str(tensor.device)


class PTRunner(object):
    """
    Runner is the replacement of PyTorch models, it caches sessions for various input desc.
    Every time tensors feed, runner will forward them to corresponding session(cache hit)
    or trigger a new session build(cache miss), then return the results.
    """
    def __init__(self, model, codegen_flags=None, **kwargs):
        """
        Parameters:
            model: torch.nn.Module to be converted.
            codegen_flags: NNFusion codegen flags, 
                ref: https://github.com/microsoft/nnfusion/wiki/4.3-NNFusion-CLI-Interface#cli-flags
            kwargs: arguments for sessions
        """
        self._model = model
        self._sessions = defaultdict(dict)
        self._codegen_flags = codegen_flags or {}
        self._session_kwargs = kwargs

    def _retrieve_by_tensor(self, tensors):
        raise NotImplementedError()

    def _retrieve_by_desc(self, descs, device):
        if device not in self._sessions[tuple(descs)]:
            self._sessions[tuple(descs)][device] = PTSession(
                self._model,
                descs,
                device,
                codegen_flags=self._codegen_flags,
                **self._session_kwargs)
        return self._sessions[tuple(descs)][device]

    def __call__(self, *args, **kwargs):
        return self.run_by_nnf(*args, **kwargs)

    def run_by_nnf(self, *args, **kwargs):
        """
        Parameters:
            args: a list of input tensors for origin PyTorch model.
        
        Returns:
            a list of PyTorch tensors executed by NNFusion,
            they should be the same as origin PyTorch model forward results.
        """
        ## Cannot support kwargs because they have no fixed args index,
        ## but torch exportor requires a fixed sequence
        ## todo: partially support such model by `inspect`
        if len(kwargs) != 0:
            raise Exception("Model forward with kwargs not supported yet")
        descs, devices = zip(*(extract_desc_and_device("input_{}".format(i), v)
                               for i, v in enumerate(args)))
        unique_devices = set(devices)
        if len(unique_devices) != 1:
            raise Exception(
                "All input tensors should be on the same device: {}".format(
                    unique_devices))
        device = list(unique_devices)[0]
        feeds = {"input_{}".format(i): v for i, v in enumerate(args)}
        return self._retrieve_by_desc(descs, device)(feeds)
