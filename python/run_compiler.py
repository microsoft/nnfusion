from memopt.engine import Engine, load_model, save_results
import memopt
import arch
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default="")
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--arch', type=str, default="V100")
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()
    memopt.set_log_level(args.verbose)
    assert args.input.endswith(".json")
    start_time = time.time()
    ordered_nodes = load_model(args.input)
    engine = Engine(args.topk, arch.__getattribute__(args.arch)())
    fusion_groups = engine.run(ordered_nodes)
    if args.output != "":
        save_results(fusion_groups, args.output)
    print("Total run time: ", time.time() - start_time)

