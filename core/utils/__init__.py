from typing import Dict, List
import logging


def beautify(dictionary: Dict) -> str:
    return [f'{key}\t: {val}' for key, val in dictionary.items()]


def device(tensors: List, gpu: bool = False, numpy: bool = False):
    def _to_device(tensors: List, device: str):
        assert device in ["cuda", "cpu"]
        return [getattr(tensor, device)() for tensor in tensors]

    _device = "cuda" if gpu else "cpu"
    _tensors = _to_device(tensors, _device)
    if numpy:
        _tensors = [tensor.numpy() for tensor in _tensors]
    return tuple(_tensors)
