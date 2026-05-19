import torch


def _cuda_capability():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}{minor}"
    return "120"


def MyArch():
    cap = _cuda_capability()
    return ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]


def AllArch():
    return MyArch()


def GetArchs():
    return MyArch()
