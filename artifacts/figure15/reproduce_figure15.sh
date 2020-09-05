# run all baselines and print log
bash codegen_and_build.sh
bash run_all.sh

# analyze logs and generate dat files
python process_log.py

# draw figure
# paper result
cd paper_result/
gnuplot gpu1_interplay_cuda_multifig.plt
cd ..
# reproduce result
cd reproduce_result/
gnuplot gpu1_interplay_cuda_multifig.plt
cd ..