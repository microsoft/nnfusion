import tensorflow as tf
import numpy as np
import argparse
import os
import time
import os.path as osp
import tempfile

def load_graph(onnx_file):
    import onnx
    from onnx_tf.backend import prepare
    onnx_model = onnx.load(onnx_file)
    tf_rep = prepare(onnx_model, device="cuda")
    exported = tempfile.TemporaryDirectory()
    tf_rep.export_graph(exported.name)
    return exported

def run_tf(prefix, xla=False):
    if xla:
        tf.config.optimizer.set_jit(True)
    exported = load_graph(osp.join(prefix, "model.onnx"))
    feed_dict = dict(np.load(osp.join(prefix, "inputs.npz"), allow_pickle=True))

    saved_model_loaded = tf.saved_model.load(
        exported.name, tags=[tf.saved_model.SERVING])
    graph_func = saved_model_loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def get_runtime():
        tic = time.time()
        _ = graph_func(**feed_dict)
        return (time.time() - tic) * 1000
    _ = [get_runtime() for i in range(200)] # warmup
    times = [get_runtime() for i in range(800)]
    print(np.mean(times), np.min(times), np.max(times))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xla', action="store_true")
    parser.add_argument('--prefix', type=str, default="temp")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    run_tf(args.prefix, xla=args.xla)
