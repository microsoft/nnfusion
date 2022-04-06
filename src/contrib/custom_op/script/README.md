# Adding Script-based Custom Operator in NNFusion
[Read this about custom op](https://github.com/microsoft/nnfusion/blob/master/src/contrib/custom_op/README.md)   
Script based custom op has same data interface with normal JSON custom op interface. Currently we've tested the inferface for antares and HLSL.

## How to add script-based custom operator?
Firstly, register custom operator in $(NNFUSION_HOME)/contrib/custom_op/script/<any_file>.json, e.g. in op.json, using below code.
```JSON
{ "ops":[
  {
    "op": "TopK",
    "script" : "python <NNFUSION_HOME>/custom_op/script --operator-name=\"<OP_NAME>\" --input-config=<OP_JSON>"
  }]
}
```
NNFusion will try to pass all config from an unkown operator trough standard input to the script and retrieve the result from standard output of the script.
The script will have not to be $(NNFUSION_HOME)/custom_op/script/\_\_main\_\_.py. It could be any script accept JSON as input and write back in JSON.
The input for the script will contain all attributes, constant input and all inputs' shape with data type which is imported from the frontend.
```JSON
{
  "axis":1,
  "largest":1,
  "sorted":1,
  "K":3,
  "input":{
    "shape":[
      [3,4]
    ],
    "dtype":[
      "float32"
    ]
  },
}
```
The output expected from the script should have 1. output shape and data type if you want do shape/dtype inference using the script; 2. antares ir if you want to use antares. 
3. HLSL kernel code if you want to insert HLSL custom kernel. An legal output is like this:
```JSON
{
  "axis":1,
  "largest":1,
  "sorted":1,
  "K":3,
  "input":{
    "shape":[
      [3,4]
    ],
    "dtype":[
      "float32"
    ]
  },
  "output":{
    "shape":[
      [3,3],
      [3,3]
    ],
    "dtype":[
      "float32",
      "int32"
    ]
  },
  "launch_config":[
    [3,1,1],
    [2,1,1]
  ],
  "entry_point":"CSMain",
  "hlsl_kernel":"<Omitted due to to long>"
}
```

## TopK as HLSL example

See [TopK.py](https://github.com/microsoft/nnfusion/blob/wenxh/topk_doc/src/contrib/custom_op/script/TopK.py) for detail source code.   
1. Register TopK script kernel  in script/op.json. [Link](https://github.com/microsoft/nnfusion/blob/bc2fbb17594e4b10552e7745c0c28222e4b0a8b8/src/contrib/custom_op/script/op.json#L8)
2. When NNFusion invoke the script module using python,  it will load the class in this folder if the base is OperatorBase and has the same name as the operator. [Link](https://github.com/microsoft/nnfusion/blob/bc2fbb17594e4b10552e7745c0c28222e4b0a8b8/src/contrib/custom_op/script/__operator__.py#L100)
3. The script will pass the input JSON and do shape/data type inference.[Link](https://github.com/microsoft/nnfusion/blob/bc2fbb17594e4b10552e7745c0c28222e4b0a8b8/src/contrib/custom_op/script/TopK.py#L109)
4. The script will generate HLSL kernel afterwards.[Link](https://github.com/microsoft/nnfusion/blob/bc2fbb17594e4b10552e7745c0c28222e4b0a8b8/src/contrib/custom_op/script/TopK.py#L52)


If you want to test the customized kernel, you can implement the OperatorTestBase class for the Kernel. And test it by the tool we'll provide soon.

There must not have any "antares_ir" field in the output JSON, since it will overwrite the HLSL kernel.

## GridSample as AntaresIR example
See [GridSample.py](https://github.com/microsoft/nnfusion/blob/master/src/contrib/custom_op/script/GridSample.py) for detail source code.
1. Register the script kernel as TopK.
2. Do shape inference without Antares.[Link](https://github.com/microsoft/nnfusion/blob/bc2fbb17594e4b10552e7745c0c28222e4b0a8b8/src/contrib/custom_op/script/GridSample.py#L32)
3. Attach antares_ir into the output.[Link](https://github.com/microsoft/nnfusion/blob/bc2fbb17594e4b10552e7745c0c28222e4b0a8b8/src/contrib/custom_op/script/GridSample.py#L12)

## Template
We've provided two template for customized operator, see: [TemplateAntaresOperator.py](https://github.com/microsoft/nnfusion/blob/master/src/contrib/custom_op/script/TemplateAntaresOperator.py), [TemplateHLSLOperator.py](https://github.com/microsoft/nnfusion/blob/master/src/contrib/custom_op/script/TemplateHLSLOperator.py) 