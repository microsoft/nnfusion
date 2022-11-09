if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser( prog = 'Test with ONNX Test cases')
    parser.add_argument('-n', '--name', default="abs,acos") 
    parser.add_argument('-f', '--file', default="default_operators.txt") 
    parser.add_argument('-m', '--mode', default="name") 
    args = parser.parse_args()
    if args.mode == "file":
        f = open(args.file).readlines()
        oplist = [v.strip().lower() for v in f]
        for op in oplist:
            os.system('python test/python/onnx_test.py --mode=name --name="'+op+'" >> log.txt')