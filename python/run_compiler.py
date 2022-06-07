from memopt.engine import run, load_model, save_results
import memopt
import arch
import argparse

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
    ordered_nodes = load_model(args.input)
    fusion_groups = run(
        ordered_nodes,
        args.topk,
        arch=arch.__getattribute__(args.arch)()
    )
    if args.output != "":
        save_results(fusion_groups, args.output)

