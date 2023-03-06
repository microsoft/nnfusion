import os
import sys
import json
import socket
import argparse
import subprocess
from contextlib import contextmanager
from wslpath import wslpath
# pip install wslpath-python


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def run(exec_path, port = 65432, host = '127.0.0.1'):
    cmd_options = [
        '-f onnx',
        '-p "batch_size:1"',
        '-fmulti_shape=false',
        '-fort_folding=false',
        '-fdefault_device=HLSL',
        '-fhlsl_codegen_type=cpp',
        '-fantares_mode=true',
        '-fblockfusion_level=0',
        '-fkernel_fusion_level=0',
        '-fkernel_tuning_steps=0',
        '-ffold_where=0',
        '-fsymbolic=0',
        '-fsplit_softmax=0',
        '-fhost_entry=0',
        '-fir_based_fusion=1',
        '-fextern_result_memory=1',
        '-fuse_cpuprofiler=1',
        '-ftuning_platform="win64"',
        '-fantares_codegen_server=127.0.0.1:8880',
    ]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        while True:
            conn, addr = s.accept()
            with conn:
                print(f'Connected by {addr}')
                while True:
                    data = conn.recv(10240)
                    if not data:
                        break
                    params = data.decode()
                    ret = {
                        'ret' : True,
                        'error' : '',
                    }
                    try:
                        params = json.loads(params)
                        model_path = wslpath(params['model'])
                        output_dir = wslpath(params['output_dir'])
                        with cd(output_dir):
                            cmd = ' '.join([exec_path, model_path] + cmd_options)
                            out = subprocess.run(cmd, stderr = subprocess.STDOUT, shell = True, encoding = 'utf8')
                        if out.returncode != 0:
                            ret['ret'] = False
                            ret['error'] = out.stderr
                        print('model_path:', model_path)
                        print('output_dir:', output_dir)
                        print('return code:', out.returncode)
                        print('stdout:', out.stdout)
                        print('stderr:', out.stderr)
                    except Exception as e:
                        print(e)
                        ret['ret'] = False
                        ret['error'] = str(e)
                    conn.sendall(bytes(json.dumps(ret), 'utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('NNFusion WSL Codegen Server')
    parser.add_argument('exec_path', type = str, help = 'path to nnfusion executable')
    parser.add_argument('--port', type = int, default = 65432, help = 'comunication port between WSL and host')
    args = parser.parse_args()
    run(os.path.abspath(args.exec_path), args.port)