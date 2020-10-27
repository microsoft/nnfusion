source ../.profile

rm -rf logs
mkdir logs



# run rammer_base_fast
cd rammer_base_fast/

cd resnext_nchw_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs1.rammerbase.fast.1000.log 2>&1
cd ../..

cd resnext_nchw_bs4_rammerbase/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs4.rammerbase.fast.1000.log 2>&1
cd ../..

cd lstm_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/lstm_bs1.rammerbase.fast.1000.log 2>&1
cd ../..

cd lstm_bs4_rammerbase/cuda_codegen
./main_test > ../../../logs/lstm_bs4.rammerbase.fast.1000.log 2>&1
cd ../..

cd ..

# run rammer_fast
cd rammer_fast/

cd resnext_nchw_bs1_rammer/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs1.rammer.fast.1000.log 2>&1
cd ../..

cd resnext_nchw_bs4_rammer/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs4.rammer.fast.1000.log 2>&1
cd ../..

cd lstm_bs1_rammer/cuda_codegen
./main_test > ../../../logs/lstm_bs1.rammer.fast.1000.log 2>&1
cd ../..

cd lstm_bs4_rammer/cuda_codegen
./main_test > ../../../logs/lstm_bs4.rammer.fast.1000.log 2>&1
cd ../..

cd ..


# run rammer_base_select
cd rammer_base_select/

cd resnext_nchw_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs1.rammerbase.select.1000.log 2>&1
cd ../..

cd resnext_nchw_bs4_rammerbase/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs4.rammerbase.select.1000.log 2>&1
cd ../..

cd lstm_bs1_rammerbase/cuda_codegen
./main_test > ../../../logs/lstm_bs1.rammerbase.select.1000.log 2>&1
cd ../..

cd lstm_bs4_rammerbase/cuda_codegen
./main_test > ../../../logs/lstm_bs4.rammerbase.select.1000.log 2>&1
cd ../..

cd ..

# run rammer_select
cd rammer_select/

cd resnext_nchw_bs1_rammer/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs1.rammer.select.1000.log 2>&1
cd ../..

cd resnext_nchw_bs4_rammer/cuda_codegen
./main_test > ../../../logs/resnext_nchw_bs4.rammer.select.1000.log 2>&1
cd ../..

cd lstm_bs1_rammer/cuda_codegen
./main_test > ../../../logs/lstm_bs1.rammer.select.1000.log 2>&1
cd ../..

cd lstm_bs4_rammer/cuda_codegen
./main_test > ../../../logs/lstm_bs4.rammer.select.1000.log 2>&1
cd ../..

cd ..