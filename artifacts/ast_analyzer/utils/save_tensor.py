def save_tensor_bin(path, tensor):
    with open(f"{path}.bin", "wb") as f:
        tensor.cpu().numpy().tofile(f)
    with open(f"{path}.shape", "w") as f:
        f.write(" ".join([str(x) for x in tensor.shape]))
