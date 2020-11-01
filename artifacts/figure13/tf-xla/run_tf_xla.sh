echo "run tf-gpu, and tf-xla"

cd ../../models/
cd resnext_imagenet_nchw
python resnext_inference.py --num_iter 1000 > ../../figure13/logs/resnext_imagenet_nchw_bs1.tf.1000.log 2>&1
python resnext_inference.py --num_iter 1000 --xla True > ../../figure13/logs/resnext_imagenet_nchw_bs1.xla.1000.log 2>&1
cd ..

cd nasnet_imagenet_nchw
python nasnet_imagenet_inference.py --num_iter 1000 > ../../figure13/logs/nasnet_imagenet_nchw_bs1.tf.1000.log 2>&1
python nasnet_imagenet_inference.py --num_iter 1000 --xla True > ../../figure13/logs/nasnet_imagenet_nchw_bs1.xla.1000.log 2>&1
cd ..
