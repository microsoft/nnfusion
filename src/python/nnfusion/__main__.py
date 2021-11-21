# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys, os, tarfile

def main():
    this_folder = os.path.split(os.path.realpath(__file__))[0]
    nnf_bin=os.path.join(this_folder, "bin/nnfusion")
    nnf_dir=os.path.join(this_folder, "bin/")
    nnf_pkg = os.path.join(this_folder, "nnfusion.tar.gz")

    if not os.path.exists(nnf_bin):
        if os.path.exists(nnf_pkg):
            tar = tarfile.open(nnf_pkg, 'r:gz')
            tar.extractall(nnf_dir)
            tar.close()
            if not os.path.exists(nnf_bin):
                print("Corrupted nnfusion.tar.gz: Please reinstall nnfusion python package.")
                exit(-1)
        else:
            print("Corrupted nnfusion.tar.gz: Please reinstall nnfusion python package.")
            exit(-1)
    args = " ".join(sys.argv[1:])
    os.system("%s %s"%(nnf_bin, args))

if __name__ == '__main__':
    main()