## How to add a new flag

1. Include the object into your cmake command: 

   `target_link_library(your_object gflags)`
2. Include the head file `"gflags/gfglas.h"`:

   This is included in `"nnfusion_common.h"`, so mostly you don't have to include this mannually.
3. `DEFINE_bool(NAME, DEFUALT_VAL,"DESCRIPTION")`;

   Use this macro to defene a flag, this flag can be assigned value in CLI command. 
   Here are other functions can be used: `DEFINE_int32/int64/uint64/double/string …`;
4. `DECLARE_bool(NAME);`

   If the flag is defined in other cpp file, you can use this to declare an exsited flag.
5. `Bool flag = FLAGS_NAME;  `

   Remember the affix of "FLAGS_"

## How to name a new flag

(Reference: GCC Options), not forced to follow
|Categories	|affix|	Example|
|-|-|-|
|Warning	|-W|	
|Optimization	|-f	|-fkernel_tunning
|Debugging	|-g||	
|Machine-dependent	|-m||	
|Other	|…|	As your will|


## CLI Flags
### Frontend
|Name|Default|Message|
|-|-|-|
|-format, -f|tensorflow|Model file format (tensorflow(default) or torchscript, onnx)|
|-params, -p|"##UNSET##"|Model input shape and type, fot torchscript, it's full shape like \"1,1:float;2,3,4,5:double\", for onnx, it's dynamic dim like \"dim1_name:4;dim2_name:128\"|

### Kernels
|Name|Default|Message|
|-|-|-|
|-frocm_fixed_kernels|True|Enable Fixed kernel in ROCm codegen.|
|-frocm_candidate_kernels|True|Enable some candidate kernels in ROCm.|

### Engine
|Name|Default|Message|
|-|-|-|
|-fdefault_device|CUDA|Choose defualt device from [CUDA, CPU, ROCm, HLSL] in the codegen.|

### Utility
|Name|Default|Message|
|-|-|-|
|-min_log_level|0|Logging level: 0 = DEBUG; 1 = INFO; 2 = WARNING; 3 = ERROR; 4 = FATAL.|

### Pass
|Name|Default|Message|
|-|-|-|
|-fcodegen_debug |false| Add debug functions in Codegen-ed project.|
|-fcodegen_timing|false| Add timing functions in Codegen-ed project.
|-fadd_allreduce|false|Add Allreduce operater after ApplyGradient operator.
|-fkernel_fusion_level|2|0: no fuse; 1: fuse element kernels; 2: fuse elem+broadcast+reshape; 3: split independent groups|
|-ffold_reshape_op|true|Folding Reshape operators.
|-fconst_folding_backend|""|Choose which backend will be used in Constantfolding pass. Disable when not set.
|-ftranspose_vecdot|false|Enable vectdot transpose.
|-fkernel_selection|true|Select kernel before codegen.
|-fkernel_tunning|false|Tunning and choose best kernel when do kernel selection.
|-frt_const_folding|false|Add runtime constant folding.
|-fmem_trace|false|Record and dump memory trace
|-fmem_log_path|memory.log|The file path of memory log.
|-fnum_stream|1|Number of streams.
|-fnuma_node_num|1|Number of numa_node.
|-fthread_num_per_node|CPU Cores / numa_node_num|Thread num of per node.
|-fantares_codegen_server|""|Antares codegen server address and port, format: \<ip\>:\<port\>
|-fnum_non_cpu|1|Number of devices.
|-fkernels_as_files|false|Saving kernels as standalone source code files.
|-fkernels_files_number|-1|Saving kernels into how many source code files.
|-fuse_default_stream|true|Use default stream.
|-fcuda_init_stream|default|The stream of kernels in cuda_init().
|-fstream_assign_policy|naive|Choose stream-assign policy from [naive, kernel_prof_based].
|-fpara_json_file|./para_info.json|Kenel entry parameter info json file.
|-ftraining_mode|false|Turn on training mode.
|-fextern_result_memory|false|Model result tensor memory is managed externally.
|-fenable_kernel_profiling|false|Profile kernel time cost.
|-fmerge_prof_compiling|false|
|-fautodiff|false|Add backward graph.
|-fantares_mode|false|Enable antares mode.
|-fcse|true|Common subexpression elimination.
|-fpattern_substitution|true|Substitute listed patterns with more efficient implementations.










