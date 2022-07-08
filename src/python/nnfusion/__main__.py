# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import site
import subprocess
import sys

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def run_cli():
    nnf_bin = os.path.join(site.USER_BASE, "share/nnfusion/nnfusion")
    if not os.path.exists(nnf_bin):
        nnf_bin = os.path.join(sys.prefix, "share/nnfusion/nnfusion")
    if not os.path.exists(nnf_bin):
        logging.error("No nnfusion cli found: Try to reinstall nnfusion.")
        sys.exit(-1)

    cmd = " ".join([nnf_bin] + sys.argv[1:])
    return subprocess.call(cmd, shell=True)

def init_env():
    if "NNFUSION_HOME" not in os.environ:
        os.environ["NNFUSION_HOME"] = os.path.join(os.path.expanduser('~'), ".nnfusion")
    logging.info("$NNFUSION_HOME is set as " + os.environ["NNFUSION_HOME"])
    if "NNFUSION_CONTRIB" not in os.environ:
        nnf_contrib = os.path.join(sys.prefix, "share/nnfusion")
        if not os.path.exists(nnf_contrib):
            nnf_contrib = os.path.join(os.environ["NNFUSION_HOME"], "contrib")
        os.environ["NNFUSION_CONTRIB"] = nnf_contrib
    logging.info("$NNFUSION_CONTRIB is set as " + os.environ["NNFUSION_CONTRIB"])

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
    init_env()
    return run_cli()


if __name__ == '__main__':
    sys.exit(main())