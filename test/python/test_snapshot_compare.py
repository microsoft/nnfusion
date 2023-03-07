import os

def compare(a, b):
    a_result = open(a).readlines()
    a_res_dict = dict()
    for l in a_result:
        a_res_dict[l.split(",")[2].strip()] = l
    b_result = open(b).readlines()
    b_res_dict = dict()
    for l in b_result:
        b_res_dict[l.split(",")[2].strip()] = l
    
    for k in a_res_dict:
        a_line = a_res_dict[k].split(",")
        if k not in b_res_dict.keys():
            print(a_res_dict[k].strip())
            print("vs ground truth:")
            print(b_res_dict[k])
        b_line = b_res_dict[k].split(",")
        if a_line[-1].strip() != b_line[-1].strip():
            print(a_res_dict[k].strip())
            print("vs ground truth:")
            print(b_res_dict[k])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( prog = 'Compare test result')
    parser.add_argument('-t', '--test_result', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_snapshot/default.csv")) 
    parser.add_argument('-g', '--ground_truth', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth/default.csv")) 
    args = parser.parse_args()
    compare(args.test_result, args.ground_truth)