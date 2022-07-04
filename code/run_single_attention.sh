#!/bin/sh
#export BERT_PATH=/content/drive/MyDrive/NEZHA-Base-WWM
#export BERT_PATH=/content/drive/MyDrive/colab_syzong/pretrain_model/chinese_roberta_wwm_ext_L-12_H-768_A-12
#export BERT_PATH=//content/drive/MyDrive/colab_syzong/pretrain_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
#export BERT_PATH=/home/syzong/nlp_deeplearning/chinese_roberta_wwm_ext_L-12_H-768_A-12
export BERT_PATH=/home/syzong/nlp_deeplearning/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
#export BERT_PATH=/home/syzong/nlp_deeplearning/NEZHA-Large-WWM
#export BERT_PATH=/home/syzong/nlp_deeplearning/NEZHA-Base-WWM
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
export PATH=./:$PATH
export TYPE=bert

export POS_MODEL=0

python -u run_nezha_atttion.py \
  --do_train=true \
  --pos_model=$POS_MODEL \
  --model_type=$TYPE \
  --data_dir=./data \
  --vocab_file=$BERT_PATH/vocab.txt \
  --bert_config_file=$BERT_PATH/bert_config.json \
  --init_checkpoint=$BERT_PATH/bert_model.ckpt \
  --max_seq_length=96 \
  --train_batch_size=64 \
  --k_fold=5 \
  --learning_rate=2e-5 \
  --num_train_epochs=5 \
  --output_dir=./bert_attention/$POS_MODEL/bert_output \
  --model_save=./bert_attention/$POS_MODEL/bert_output/model

for value in 0 1 2 3 4
do
     echo "Now value is $value..."
     python run_nezha_atttion.py \
        --do_predict=true \
        --pos_model=$POS_MODEL \
        --model_type=$TYPE \
        --data_dir=./data \
        --vocab_file=$BERT_PATH/vocab.txt \
        --bert_config_file=$BERT_PATH/bert_config.json \
        --init_checkpoint=./bert_attention/$POS_MODEL/bert_output-fold-$value/model \
        --max_seq_length=96 \
        --output_dir=./bert_attention/$POS_MODEL/bert_output-fold-$value
done

