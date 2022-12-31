#!/bin/bash
# 
#CompecTA (c) 2018
# 
# You should only work under the /scratch/users/<username> directory.
#
# Jupyter job submission script
#
# TODO:
#   - Set name of the job below changing "JupiterNotebook" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have maximum run time of 1 day..
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch jupyter_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=Tuning
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --mem=32G
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --time=1-00:00:00
#SBATCH --output=tuning-%J.log




# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
module load cuda/10.2
module load cudnn/8.1.1/cuda-10.2
module load anaconda/2022.5
conda activate cartography

echo "======================="

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################




# DON'T USE ADDRESS BELOW. 
# DO USE TOKEN BELOW
python -m src.main -mode train -project_name test_runs -hidden_size 512 -depth 3 -model_selector_set val -pretrained_model_name none -finetune_data_voc none -dev_set -no-test_set -no-gen_set -dataset cfq/train_mcd1_1 -dev_always -no-test_always -no-gen_always -epochs 100 -save_model -show_train_acc -embedding random -no-freeze_emb -no-freeze_emb2 -no-freeze_lstm_encoder -no-freeze_lstm_decoder -no-freeze_fc -batch_size 256 -lr 0.001 -emb_lr 0.0005 -dropout 0.1 -no_beam_decode -early_stopping 50 -run_name cfq_mcd1_seq2seq_1 -gpu 0 -topk 1
# todo change epoch
