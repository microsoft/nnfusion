from tvm import te

def tvm_matmul(n, m, k):
    A = te.placeholder((m, k), name="A")
    B = te.placeholder((k, n), name="B")
    k = te.reduce_axis((0, k), name="k")
    C = te.compute((m, n), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k))
    s = te.create_schedule(C.op)
    return s, (A, B, C)
