#!python
import json
import sys
import argparse
from __operator__ import get_operator_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='General Shape Inference Tool for NNFusion Contributed Kernel.')
    parser.add_argument('--operator-name', metavar='string',
                        type=str, help='Operator name')
    parser.add_argument('--input-config', metavar='json-string',
                        type=str, help='Valid JSON string of shape')
    args = parser.parse_args()

    input_dict = json.loads(args.input_config)
    print(json.dumps(get_operator_config(args.operator_name, input_dict)))
