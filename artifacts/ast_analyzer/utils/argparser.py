import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--profile', type=str, default='off', choices=['off', 'sys', 'pytorch', 'torchscript'])
parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'])

parser.add_argument('--sys', dest='run_sys', action='store_true')
parser.add_argument('--no-sys', dest='run_sys', action='store_false')
parser.set_defaults(run_sys=True)

parser.add_argument('--torch', dest='run_pytorch', action='store_true')
parser.add_argument('--no-torch', dest='run_pytorch', action='store_false')
parser.set_defaults(run_pytorch=True)

parser.add_argument('--sct', dest='run_sct', action='store_true')
parser.add_argument('--no-sct', dest='run_sct', action='store_false')
parser.set_defaults(run_sct=False)

parser.add_argument('--step', dest='profile_step', action='store_true')
parser.add_argument('--no-step', dest='profile_step', action='store_false')
parser.set_defaults(profile_step=False)

parser.add_argument('--measure', dest='measure', action='store_true')
parser.add_argument('--no-measure', dest='measure', action='store_false')
parser.set_defaults(measure=False)

parser.add_argument('--enable-cf', dest='cf', action='store_true')
parser.add_argument('--disable-cf', dest='cf', action='store_false')
parser.set_defaults(cf=True)

parser.add_argument('--platform', type=str, default='V100', choices=['V100', 'MI100'])

parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)

parser.add_argument('--enable-breakdown', dest='breakdown', action='store_true')

def get_parser():
    return parser