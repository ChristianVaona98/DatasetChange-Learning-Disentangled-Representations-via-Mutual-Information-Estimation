# DatasetChange-Learning-Disentangled-Representations-via-Mutual-Information-Estimation

# Learning Disentangled Representations via Mutual Information Estimation

N.B.: Extract the data file .zip in the 'data/SmallNORB' folder before running the project!

# Training commands
## Start shared representation training

	echo "Start shared representation training"
	# Set environment variables
	$data_base_folder = "data"
	$xp_name = "Share_representation_training"
	$conf_path = "conf/share_conf.yaml"
	$seed = 42

	$env:PYTHONPATH = "$env:PYTHONPATH;src"
	python src/sdim_train.py --xp_name $xp_name --conf_path $conf_path --data_base_folder $data_base_folder --seed $seed

## Start exclusive representation training
	echo "Start exclusive representation training"
	# Set environment variables
	$data_base_folder = "data"
	$xp_name = "Exclusive_representation_training"
	$conf_path = "conf/exclusive_conf.yaml"
	$seed = 42
	$trained_enc_x_path = "mlruns/1/984263f802d040f88aa4f9b295e7fe57/artifacts/sh_encoder_x/state_dict.pth"
	$trained_enc_y_path = "mlruns/1/984263f802d040f88aa4f9b295e7fe57/artifacts/sh_encoder_y/state_dict.pth"

	$env:PYTHONPATH = "$env:PYTHONPATH;src"
	python src/edim_train.py --xp_name $xp_name --conf_path $conf_path --data_base_folder $data_base_folder --seed $seed --trained_enc_x_path $trained_enc_x_path --trained_enc_y_path $trained_enc_y_path

# Testing commands
## Start shared representation testing
	$data_base_folder="data"
	$xp_name="Share_representation_testing"
	$conf_path="conf/share_conf.yaml"
	$seed = 42

	$PYTHONPATH=$PYTHONPATH:src 
	python src/sdim_test.py --xp_name $xp_name --conf_path $conf_path --data_base_folder $data_base_folder --seed $seed

## Start exclusive representation testing
	echo "Start exclusive representation testing"
	$data_base_folder="data"
	$xp_name="Exclusive_representation_testing"
	$conf_path="conf/exclusive_conf.yaml"
	$seed = 42
	$trained_enc_x_path="mlruns/1/f5f880f36ca94974aad06df0b050c5da/artifacts/sh_encoder_x/state_dict.pth"
	$trained_enc_y_path="mlruns/1/f5f880f36ca94974aad06df0b050c5da/artifacts/sh_encoder_y/state_dict.pth"

	$PYTHONPATH=$PYTHONPATH:src 
	python src/edim_test.py --xp_name $xp_name --conf_path $conf_path --data_base_folder $data_base_folder --seed $seed --trained_enc_x_path $trained_enc_x_path --trained_enc_y_path $trained_enc_y_path


Note: Remember to change the path to the pretrained encoder of domains X and Y. I have also added a seed = 42 to make the run reproducible

