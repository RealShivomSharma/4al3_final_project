import torch


def test():

    if not (torch.backends.mps.is_available() or torch.backends.cuda.is_built()):
        raise RuntimeError("No working backend found for torch on this system.")

    if torch.backends.mps.is_available():
        print("Backend Running MPS for macOS")

    if torch.backends.cuda.is_built():
        print("Backend Running CUDA for windows")

    return


if __name__ == "__main__":

    test()
