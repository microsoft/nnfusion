from memopt.engine import Engine, load_model, save_results
import memopt
import arch
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str, default="")
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--arch', type=str, default="V100")
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--check', action="store_true")
    args = parser.parse_args()
    memopt.set_log_level(args.verbose)
    assert args.input_file.endswith(".json")
    start_time = time.time()
    ordered_nodes = load_model(args.input_file)
    engine = Engine(args.topk, arch.__getattribute__(args.arch)(),
        device="cuda:{}".format(args.device), enable_checking=args.check)
    fusion_groups = engine.run(ordered_nodes)
    gain = sum([fg.gain for fg in fusion_groups])
    print("Fusion gain: {}ms".format(gain))
    if args.output_file != "":
        save_results(fusion_groups, args.output_file)
    print("Total run time: ", time.time() - start_time)

