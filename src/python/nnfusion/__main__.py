# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import site
import subprocess
import sys

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def run_cli():
    nnf_bin = os.path.join(site.USER_BASE, "nnfusion-bin/nnfusion")
    if not os.path.exists(nnf_bin):
        nnf_bin = os.path.join(sys.prefix, "nnfusion-bin/nnfusion")
    if not os.path.exists(nnf_bin):
        logging.error("No nnfusion cli found: Try to reinstall nnfusion.")
        sys.exit(-1)

    cmd = " ".join([nnf_bin] + sys.argv[1:])
    return subprocess.call(cmd, shell=True)


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
    return run_cli()


if __name__ == '__main__':
    sys.exit(main())
