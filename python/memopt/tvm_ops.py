from tvm import te

def tvm_matmul(n, m, k):
    A = te.placeholder((n, k), name="A")
    B = te.placeholder((k, m), name="B")
    k = te.reduce_axis((0, k), name="k")
    C = te.compute((n, m), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k))
    return (A, B, C)
