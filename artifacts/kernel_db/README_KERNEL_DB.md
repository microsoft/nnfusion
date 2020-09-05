# Kernel DB for RammerBase and Rammer

The kernel_cache.db file is the generated kernel DB.

## Generate Kernel DB for RammerBase and Rammer

This script (reproduce_kernel_db.sh) will leverage TVM-0.7 to generate kernels with AutoTVM logs (autotvm_logs folder), and import kernels into the kernel DB (the kernel_cache.db file will be replaced). 

This script is used to generate all the kernels used by Rammer and RammerBase in our evaluation. 

The result kernels will be injected in to a kernel database, localed in both *~/.cache/nnfusion/kernel_cache.db* and  *./kernel_cache.db*, which will be finally loaded by NNFusion.

**NOTE**: this process will take about 1 hour :)
```bash
cd ~/nnfusion/artifacts/kernel_db/
bash reproduce_kernel_db.sh
```