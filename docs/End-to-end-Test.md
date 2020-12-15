The new e2e testing tool is targeting easy usage and could be easily extented.

The tool is located at "(source code repo)/test/nnfusion/script/e2e_test.py".

Execution in native enviroment:

**1. Preparation**:

   Download all the models & test description JSON's:

   `./maint/script/download_models.sh`
		
   This file will use "az" to pull all the models from our shared file storage service: You can add test cases to "(nnfusion file share)/../frozenmodels/pipeline".
	
**2. Write a config to guide your test:**

   A basic config file is located at "./test/nnfusion/script/config.json":

```json
{
"env": {
"HSA_USERPTR_FOR_PAGED_MEM": {
"set": 0
},
"LD_LIBRARY_PATH": {
"append": "/usr/local/lib"
},
"HIP_VISIBLE_DEVICES" : {
"set": 1
}
},
"device_capability": ["CUDA", "ROCM"],
"enabled_tags": {
"CUDA" : ["correctness", "correctness_cuda_only"],
"ROCM" : ["correctness", "correctness_rocm_only"]
},
"models": "~/models",
"nnfusion_cli": "/usr/local/bin/nnfusion"
}

```	
   In this file, you can use `"set"`, `"append"`, `"clear"` to control the enviorment variable;

   Use `"device_capability"` to specify on which device you want to run test;

   And use `"enabled_tags"` to specify which group(under the tags) need to be run;
	
**3. Add test cases:**

   After run `download_models.h`, a folder named "frozenmodels" is created and all the required models are downloaded inside this folder.  
	
   A test config can be added and loaded by default test run under the folder "(source root)/test/nnfusion/scripts/testcase_configs".
	 
   A  test config is shown here:

```json
{
"testcases": [
{
"testcase": "naive_cases #1",
"tag": [
"correctness"
],
"filename": "pipline/naïve_tests/frozen_random-weights_bert_large.pb",
"type": "naive_case_single_line",
"flag" : "",
"output": "0.001335 0.001490 0.000675 0.002558 0.000761 0.001435 0.000518 0.001516 0.000738 0.001183  .. (size = 1001, ends with 0.000281);",
"comment": "Naive json descriptor for test case."
},  ]
}
```
 
   This is a naïve test which only compare outputs as strings. You can see the tag(group) it has is "correctness". And the type is naïve_case_single_line, which is located at "./test/nnfusion/script/testcases/naïve_cases.py". You can add more test case classes like performance test in here. And the file name should be where the model file located, which is "(source root)/../frozenmodels/pipline/naïve_tests/frozen_random-weights_bert_large.pb".
	
**4. Run the test:**

   Here are two ways to run the test:

   a. Using config.json

  `python ./test/nnfusion/script/e2e_test.py "path to config.json"`

  If the config file is not specified, the default config will be applied.
		
   b. Use parameter

   `python  ./test/nnfusion/script/e2e_test.py "path to models folder" "path to nnfusion cli"`

   This will use default config and load test from the models folder, compiling models use the nnfusion cli specified.
		
**5. Get result:**

   The result is in the output and it looks like this:
```
   ========================================= 
   srgam-02 E2E Test report 
   ========================================= 
   ROCM naive_cases #2 correctness Succeed! 
   ROCM naive_cases #3 correctness Succeed! 
   ROCM naive_cases #4 correctness Succeed! 
   ROCM naive_cases #5 correctness Succeed! 
```	
