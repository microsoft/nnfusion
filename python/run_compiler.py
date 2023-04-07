import argparse
import time

import welder
from welder import arch
from welder.engine import (Engine, MultiProcTunner, Tunner, load_model,
                           save_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str, default="")
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--arch', type=str, default="V100")
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--check', action="store_true")
    parser.add_argument('--nofusion', action="store_true")
    args = parser.parse_args()
    welder.set_log_level(args.verbose)
    assert args.input_file.endswith(".json")
    start_time = time.time()
    ordered_nodes = load_model(args.input_file)
    # tunner = Tunner(arch=arch.__getattribute__(args.arch)(), device="cuda:{}".format(args.device), topk=args.topk, check=args.check)
    tunner = MultiProcTunner(input_file_path=args.input_file,
        arch=arch.__getattribute__(args.arch)(), device="cuda:{}".format(args.device), topk=args.topk, check=args.check)
    engine = Engine(tunner)
    if args.nofusion:
        fusion_groups = engine.run_no_fusion(ordered_nodes)
    else:
        fusion_groups = engine.run(ordered_nodes)
    gain = sum([fg.gain for fg in fusion_groups])
    print("Fusion gain: {}ms".format(gain))
    if args.output_file != "":
        save_results(fusion_groups, args.output_file)
    print("Total run time: ", time.time() - start_time)

