# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from .description import IODescription
from .session import Session


def extract_desc_and_device(name, tensor):
    return IODescription(name, tensor.shape, tensor.dtype), str(tensor.device)


class Runner(object):
    def __init__(self,
                 model,
                 external_weights=None,
                 codegen_flags=None,
                 **kwargs):
        self._model = model
        self._sessions = defaultdict(dict)
        self._codegen_flags = codegen_flags or {}
        self._external_weights = external_weights or {}
        self._session_kwargs = kwargs

    def _retrieve_by_tensor(self, tensors):
        raise NotImplementedError()

    def _retrieve_by_desc(self, descs, device):
        if device not in self._sessions[tuple(descs)]:
            self._sessions[tuple(descs)][device] = Session(
                self._model,
                descs,
                device,
                external_weights=self._external_weights,
                codegen_flags=self._codegen_flags,
                **self._session_kwargs)
        return self._sessions[tuple(descs)][device]

    def __call__(self, *args, **kwargs):
        return self.run_by_nnf(*args, **kwargs)

    def run_by_nnf(self, *args, **kwargs):
        ## Cannot support kwargs because they have no fixed args index,
        ## but torch exportor requires a fixed sequence
        ## todo: partially support such model by `inspect`
        if len(kwargs) != 0:
            raise Exception("Model forward with kwargs not supported yet")
        descs, devices = zip(*(extract_desc_and_device("input{}".format(i), v)
                               for i, v in enumerate(args)))
        unique_devices = set(devices)
        if len(unique_devices) != 1:
            raise Exception(
                "All input tensors should be on the same device: {}".format(
                    unique_devices))
        device = list(unique_devices)[0]
        feeds = {"input{}".format(i): v for i, v in enumerate(args)}
        return self._retrieve_by_desc(descs, device)(feeds)
