# python -m src.main -mode train -project_name test_runs -hidden_size 64 -model_selector_set train -pretrained_model_name none -finetune_data_voc none -no-dev_set -no-test_set -no-gen_set -dataset simple_split_tsv -no-dev_always -no-test_always -no-gen_always -epochs 150 -save_model -show_train_acc -embedding random -no-freeze_emb -no-freeze_emb2 -no-freeze_lstm_encoder -no-freeze_lstm_decoder -no-freeze_fc -batch_size 256 -lr 0.008 -emb_lr 0.0005 -dropout 0.1 -run_name RUN-train_try -gpu 1
# the baseline run script -> python -m src.main_cls -mode train -project_name test_runs_cls -hidden_size 64 -model_selector_set val -pretrained_model_name none -finetune_data_voc none -dev_set -no-test_set -no-gen_set -dataset simple_split_tsv_cls -dev_always -no-test_always -no-gen_always -epochs 150 -save_model -show_train_acc -embedding random -no-freeze_emb -no-freeze_emb2 -no-freeze_lstm_encoder -no-freeze_lstm_decoder -no-freeze_fc -batch_size 256 -lr 0.008 -emb_lr 0.001 -dropout 0.1 -no_beam_decode -run_name RUN-train_try_cls -gpu 1 -topk 1 

for DEPTH in 2 3
do
	for HIDDEN in 64 128 256
	do
		python -m src.main_cls -mode train -project_name test_runs_cls -hidden_size $HIDDEN -model_selector_set val -pretrained_model_name none -finetune_data_voc none -dev_set -no-test_set -no-gen_set -dataset simple_split_tsv_cls -dev_always -no-test_always -no-gen_always -epochs 150 -save_model -show_train_acc -embedding random -no-freeze_emb -no-freeze_emb2 -no-freeze_lstm_encoder -no-freeze_lstm_decoder -no-freeze_fc -batch_size 256 -lr 0.008 -emb_lr 0.001 -dropout 0.1 -no_beam_decode -run_name RUN-train_try_cls -gpu 0 -topk 1 -depth $DEPTH
	done
done

# todo change epoch
