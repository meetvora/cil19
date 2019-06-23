import torch
from loss import Loss


class Task(object):
    attributes = {
        'setting': str(),
        'ROOT_DIR': str(),
        'IMAGE_DIR': str(),
        'MASK_DIR': str(),
        'BATCH_SIZE': int(),
        'SHUFFLE': bool(),
        'OPTIMIZER': str(),
        'OPTIM_PARAMS': dict(),
        'LOSS': str(),
        'NUM_EPOCHS': int()
    }

    def __init__(self, kwargs):
        Task.validate_assignment(kwargs)
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    @staticmethod
    def validate_assignment(kwargs):
        assert kwargs["setting"] in ["train", "test",
                                     "val"], "Unsupported setting."
        assert kwargs[
            "LOSS"] in Loss.implemented, f"{kwargs['LOSS']} Loss not implemented."
        if "OPTIMIZER" in kwargs:
            assert kwargs["OPTIMIZER"] in [
                attr for attr in dir(torch.optim) if "__" not in attr
            ], "Unsupported optimizer."

        for arg in kwargs:
            assert arg in Task.attributes, f"Invalid keyword {arg}."
            assert type(kwargs[arg]) == type(
                Task.attributes[arg]), f"Type mismatch for {arg}."
