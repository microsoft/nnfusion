
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='matmul_expr_new_regk1_full.json')
parser.add_argument('--new_file', type=str, default='matmul_expr.json')

args = parser.parse_args()

'''
the format of args.file is like
{
    (register_tile_dim_0, ...,register_tile_dim_n, warp_size) : compute time,
}

the format of args.new_file is like
{
    (register_tile_area, warp_size)
}
'''


new_db = {}
with open (args.file, 'r') as f:
    db = json.loads(f.read())
    for k in db:
        tmp = k[1:-1].split(", ")
        area = 1
        for i in range(len(tmp) - 1):
            area *= int(tmp[i])
        new_k = []
        new_k.append(str(area))
        new_k.append(tmp[-1])

        new_db["(" + ', '.join(new_k) + ")"] = db[k]

with open(args.new_file, 'w') as f:
    json.dump(new_db, f)