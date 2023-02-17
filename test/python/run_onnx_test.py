if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser( prog = 'Test with ONNX Test cases')
    parser.add_argument('-n', '--name', default="abs,acos") 
    parser.add_argument('-f', '--file', default="default_operators.txt") 
    parser.add_argument('-o', '--output', default="log.csv") 
    parser.add_argument('-m', '--mode', default="file") 
    parser.add_argument("-i", "--input_as_constant", action="store_true", default=False)
    parser.add_argument("-a", "--float_as_half", action="store_true", default=False)
    parser.add_argument("-d", "--float_as_double", action="store_true", default=False)
    args = parser.parse_args()
    if args.mode == "file":
        f = open(args.file).readlines()
        oplist = [v.strip().lower() for v in f]
        for op in oplist:
            params = [ "python",
                        "test/python/onnx_test.py",
                        "--mode=name",
                        "--name="+op,
                        "-i" if args.input_as_constant else "",
                        "-a" if args.float_as_half else "",
                        "-d" if args.float_as_double else "",
                        ">>",
                        args.output]
            os.system(" ".join(params))
