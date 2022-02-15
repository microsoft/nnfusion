# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import sys

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def run_cli():
    nnf_bin = os.path.join(sys.prefix, "nnfusion-bin/nnfusion")
    if os.path.exists(nnf_bin):
        args = " ".join(sys.argv[1:])
        os.system("%s %s" % (nnf_bin, args))
    else:
        logging.error("No nnfusion cli found: Try to reinstall nnfusion.")
        sys.exit(-1)


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
    run_cli()


if __name__ == '__main__':
    main()
