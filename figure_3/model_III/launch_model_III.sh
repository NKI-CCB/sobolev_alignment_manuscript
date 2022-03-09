# LAUNCH_MODEL_II (FIGURE 3)
# Performs the complete Sobolev Alignment procedure, including
#	- Hyper-parameters search for scVI model.
#	- Hyper-parameters search for KRR (Falkon).
# 	- Training of VAE.
#	- Alignment.

script_folder=./sobolev_alignment/
output_folder=./output/
data_folder=.
tmp_folder=/tmp/model_II/
python_folder=python

mkdir ${output_folder}
mkdir ${output_folder}/params/
mkdir ${tmp_folder}

# Hyperopt selection
${python_folder} ${script_folder}/source_hyperopt_search.py -o $output_folder -d $data_folder -c 0.1 -m 100 \
& \
${python_folder} ${script_folder}/target_hyperopt_search.py -o $output_folder -d $data_folder -c 0.1 -m 100
${python_folder} ${script_folder}/process_hyperopt.py -o $output_folder

# Train VAE
${python_folder} ${script_folder}/train_VAE.py -o $output_folder -d $data_folder -n 10000000 -t $tmp_folder -j 50;
rm -r ${tmp_folder}'*'

# Find optimal kernel hyper-parameters.
${python_folder} ${script_folder}/analysis_approx_KRR_error.py -o $output_folder -d $data_folder -n 1000000 -t $tmp_folder -j 30
${python_folder} ${script_folder}/process_KRR_results.py -o $output_folder
rm -r ${tmp_folder}/*

# Find optimal regularization parameter.
source_tmp_folder=/tmp/SM_source/
target_tmp_folder=/tmp/SM_target/
mkdir ${source_tmp_folder}
mkdir ${target_tmp_folder}
${python_folder} ${script_folder}/same_model_calibration_source.py -o $output_folder -d $data_folder -n 2500000 -t $source_tmp_folder -j 60
${python_folder} ${script_folder}/same_model_calibration_target.py -o $output_folder -d $data_folder -n 2500000 -t $target_tmp_folder -j 60
${python_folder} ${script_folder}/process_same_model_alignment.py -o $output_folder
rm -r ${source_tmp_folder}
rm -r ${target_tmp_folder}

# Sobolev Alignment.
${python_folder} ${script_folder}/taylor_feature_attribution.py -o $output_folder -d $data_folder -n 5000000 -t $tmp_folder -j 50 -i 1;
rm -r ${tmp_folder}/'*'
rm -r ${tmp_folder}