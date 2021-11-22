# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys, os, tarfile, shutil, logging
from typing_extensions import runtime

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
this_folder = os.path.split(os.path.realpath(__file__))[0]
nnf_bin=os.path.join(this_folder, "bin/nnfusion")
nnf_dir=os.path.join(this_folder, "bin/")
nnf_pkg = os.path.join(this_folder, "nnfusion.tar.gz")

def check_pkg():
    if os.path.exists(nnf_bin) and os.path.exists(nnf_pkg):
        old_time = os.path.getctime(nnf_bin)
        new_time = os.path.getctime(nnf_pkg) 
        if new_time > old_time:
            logging.info("Replacing old nnfusion cli.")
            shutil.rmtree(nnf_dir)
    else:
        if os.path.exists(nnf_bin):
            logging.info("No nnfusion cli found: Try to extract it from nnfusion.tar.gz.")


def extract_pkg():
    if os.path.exists(nnf_pkg):
        tar = tarfile.open(nnf_pkg, 'r:gz')
        tar.extractall(nnf_dir)
        tar.close()
        if not os.path.exists(nnf_bin):
            logging.error("Corrupted nnfusion.tar.gz: Please reinstall nnfusion python package.")
            exit(-1)
    else:
        logging.error("Missing nnfusion.tar.gz: Please reinstall nnfusion python package.")
        exit(-1)

def run_cli():
    if os.path.exists(nnf_bin):
        args = " ".join(sys.argv[1:])
        os.system("%s %s"%(nnf_bin, args))
    else:
        logging.error("No nnfusion cli found: Try to reinstall nnfusion.")
        exit(-1)

def welcome():
    print("     _  __ _  __ ____            _ ")
    print("    / |/ // |/ // __/__ __ ___  (_)___   ___ ")
    print("   /    //    // _/ / // /(_-< / // _ \\ / _ \\")
    print("  /_/|_//_/|_//_/   \\_,_//___//_/ \\___//_//_/")
    print("      MSRAsia NNFusion Team(@nnfusion)")
    print("    https://github.com/microsoft/nnfusion")
    print("") 


def main():
    welcome()
    check_pkg()
    extract_pkg()
    run_cli()


if __name__ == '__main__':
    main()