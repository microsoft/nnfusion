source ../.profile

# leverage TVM-0.7 to generate kernels with AutoTVM logs
bash rammer_kernel_codegen.sh

# import generated kernels into kernel DB
bash import_kernel_db.sh