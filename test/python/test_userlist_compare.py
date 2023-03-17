import os

full_files = ["default", "float16", "float64", "input_as_constant"]
full_files_dict = {}


def compare(b, s):
    b_result = open(b).readlines()
    b_res_dict = dict()
    for l in b_result:
        b_res_dict[l.split(",")[0].strip()] = [v.strip() for v in l.split(",")[1:]] if len(l.split(","))>1 else full_files
    for t in b_res_dict:
        neg_case = []
        for f in b_res_dict[t]:
            if t not in full_files_dict[f] or full_files_dict[f][t] != s:
                neg_case.append(f)
        if t!="" and len(neg_case) > 0:
            print("Error:", t, "with config[", ",".join(neg_case) , "] is not", s)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( prog = 'Compare test result')
    parser.add_argument('-t', '--test_snapshot_folder', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_snapshot/")) 
    parser.add_argument('-g', '--ground_truth', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth/user_pass_list.csv")) 
    parser.add_argument('-s', '--status', default=os.path.join("PASS"))
    args = parser.parse_args()
    for file in full_files:
        csv_file = open(os.path.join(args.test_snapshot_folder, file + ".csv")).readlines()
        full_files_dict[file] = dict()
        for l in csv_file:
            full_files_dict[file][l.split(",")[2].strip()] = l.split(",")[-1].strip()
    compare(args.ground_truth, args.status)