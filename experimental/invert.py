from tvm import arith, ir, te


def layout(i, j):
    return [i // 8, j // 8, i % 8, j % 8]

analyzer = arith.Analyzer()
i, j = te.var("i"), te.var("j")
input_iters = [i, j]
indices = layout(i, j)
iter_map_range = {i: ir.Range(0, 64), j: ir.Range(0, 64)}
iter_map_result = arith.detect_iter_map(indices, iter_map_range, check_level=arith.iter_affine_map.IterMapLevel.Bijective)
if len(iter_map_result.errors) > 0:
    print(iter_map_result.errors)
    exit(0)
output_iter = [te.var("i"+str(i)) for i in range(len(indices))]
results = arith.iter_affine_map.inverse_affine_iter_map(iter_map_result.indices, output_iter)
print(results)
