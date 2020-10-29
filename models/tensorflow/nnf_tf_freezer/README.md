# Freeze TensorFlow models

**nnf_tf_freezer** is a tool that can freeze a tensorflow model into a protobuf file. It can run tensorflow constant folding the frozen graph. To compare the performance of tensorflow with nnfusion, you could also use this tool to feed the frozen graph (a protobuf file) to tensorflow. 

## requirement
* python >= 3.6
* tensorflow-gpu == 1.14.0

## Use nnf_tf_freezer to freeze a tensorflow model

### step1:
Download [nnf_tf_freezer.py](./nnf_tf_freezer.py), and import `nnf_tf_freezer`.

### step2: 
Construct your model, and define the `inputs : List[tf.placeholder]` and `outputs : List[tf.identity]` of the model graph. For training model, we use `tf.train.GradientDescentOptimizer(learning_rate=1e-4) ` as our default optimizer. You may also pass other `optimizer ：tf.train.Optimizer` to nnf_tf_freezer. 

### step3:
Pass parameter to nnf_tf_freezer and begin to freeze.
```
freezer = nnf_tf_freezer(args.frozen_graph, args.const_folding, args.run_graph, args.xla, args.parallel, 
        args.warmup, args.num_iter, args.run_const_folded_graph, args.debug, args.is_training)
freezer.execute(inputs, outputs, optimizer)
```

## Quick Start
First you need to download the folder [nnf_tf_freezer](./). Inside the folder, `tf_freeze_graph_example.py` gives an example on how to use tf_feezer to freeze tensorflow models. It supports freezing bert, nasnet_cifar, alexnet, deepspeech2, inception3, lstm, resnet, seq2seq and vgg models.

If you want to freeze a bert inference model, run constant folding to the frozen graph, and see the performance of tensorflow on this frozen constant-folded graph, just type the following code to your terminal:

```
python3 tf_freeze_graph_example.py --model_name=bert --frozen_graph=bert.pb --const_folding --run_graph --run_const_folded_graph
```
It will generate two files under current directory:  `bert.pb` (the original version) and `bert.const_folded.pb` (the constant-folded version). And you will see the output of this model and a summary of tensorflow performance which looks like:
```
Updated:
import/dense/Softmax:0
[0.00113481 0.00113616 0.00070336 0.00122646 0.00129572 0.00110195
 0.00107207 0.00162248 0.00052593 0.00071608] ...(size= 1001 end with 0.00069333427 )
Updated:
Iteration time 2.840042 ms
Iteration time 2.985239 ms
Iteration time 3.128767 ms
Iteration time 2.961397 ms
Iteration time 3.027201 ms
Iteration time 2.962351 ms
Iteration time 2.971411 ms
Iteration time 3.184080 ms
Iteration time 2.899170 ms
Iteration time 3.120422 ms
Summary: [min, max, mean] = [2.840042, 3.184080, 3.008008] ms
```

