#!/bin/sh
#export BERT_PATH=/content/drive/MyDrive/NEZHA-Base-WWM
#export BERT_PATH=/content/drive/MyDrive/colab_syzong/pretrain_model/chinese_roberta_wwm_ext_L-12_H-768_A-12
#export BERT_PATH=//content/drive/MyDrive/colab_syzong/pretrain_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
#export BERT_PATH=/home/syzong/nlp_deeplearning/chinese_roberta_wwm_ext_L-12_H-768_A-12
#export BERT_PATH=/home/syzong/nlp_deeplearning/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
#export BERT_PATH=/home/syzong/nlp_deeplearning/chinese_macbert_large
export BERT_PATH=/home/syzong/nlp_deeplearning/NEZHA-Large-WWM
#export BERT_PATH=/home/syzong/nlp_deeplearning/NEZHA-Large-WWM

export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
export PATH=./:$PATH
export TYPE=nezha
export LAB_POS=0

for LAB_POS in 0 1 2 3 4 5
do
    python -u run_nezha_mse_single.py \
      --do_train=true \
      --lab_pos=$LAB_POS \
      --model_type=$TYPE \
      --data_dir=./data \
      --vocab_file=$BERT_PATH/vocab.txt \
      --bert_config_file=$BERT_PATH/bert_config.json \
      --init_checkpoint=$BERT_PATH/bert_model.ckpt \
      --max_seq_length=336 \
      --train_batch_size=14 \
      --k_fold=5 \
      --learning_rate=2e-5 \
      --num_train_epochs=2 \
      --output_dir=./$TYPE/$LAB_POS/bert_output \
      --model_save=./$TYPE/$LAB_POS/bert_output/model

    for value in 0 1 2 3 4
    do
         echo "Now value is $value..."
         python run_nezha_mse_single.py \
            --do_predict=true \
            --lab_pos=$LAB_POS \
            --model_type=$TYPE \
            --data_dir=./data \
            --vocab_file=$BERT_PATH/vocab.txt \
            --bert_config_file=$BERT_PATH/bert_config.json \
            --init_checkpoint=./$TYPE/$LAB_POS/bert_output-fold-$value/model \
            --max_seq_length=336 \
            --output_dir=./$TYPE/$LAB_POS/bert_output-fold-$value
    done

done

