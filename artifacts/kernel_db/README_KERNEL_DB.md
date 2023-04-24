# Kernel DB for CocktailerBase and Cocktailer

The `reproduce_kernel_db.sh` scripts will leverage AutoTVM, Ansor, Roller, and manual implementation to generate kernels. The result kernels will be injected in to a kernel database, located in *~/.cache/nnfusion/kernel_cache.db*, which will be finally loaded by NNFusion.

This folder contains the following contents:
* `*_kernels` folders: the tuning result of each source
* `db`: scripts for injecting kernels into the kernel database, adapted from [https://github.com/microsoft/nnfusion/tree/osdi20_artifact/artifacts/kernel_db/kernel_db_scripts](https://github.com/microsoft/nnfusion/tree/osdi20_artifact/artifacts/kernel_db/kernel_db_scripts)
* `roller`: the source code of Roller, adapted from [https://github.com/microsoft/nnfusion/tree/osdi22_artifact/artifacts/roller](https://github.com/microsoft/nnfusion/tree/osdi22_artifact/artifacts/roller)
* `test_config`: the TVM implementation of each operator, adapted from [https://github.com/microsoft/nnfusion/tree/osdi22_artifact/artifacts/roller/test_config](https://github.com/microsoft/nnfusion/tree/osdi22_artifact/artifacts/roller/test_config)
