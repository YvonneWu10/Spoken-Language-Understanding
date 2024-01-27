lr_list=(5e-3 1e-3 1e-4)
dropout_list=(0.1 0.2 0.3 0.5)
num_layer_rnn_list=(2 4)
encoder_cell_list=('LSTM' 'GRU' 'RNN')
word_embedding_list=('Word2vec' 'Bert' 'WWM' 'Roberta')
trainset_spoken_language_select_list=('manual_transcript' 'asr_1best' 'both')

for lr in ${lr_list[@]};do
for dropout in ${dropout_list[@]};do
for num_layer_rnn in ${num_layer_rnn_list[@]};do
for encoder_cell in ${encoder_cell_list[@]};do
for word_embedding in ${word_embedding_list[@]};do
for trainset_spoken_language_select in ${trainset_spoken_language_select_list[@]};do
python scripts/slu_baseline_f.py --lr $lr --dropout $dropout --num_layer_rnn $num_layer_rnn \
    --encoder_cell $encoder_cell --word_embedding $word_embedding --trainset_spoken_language_select $trainset_spoken_language_select \
    --device 3 --batch_size 256
done
done
done
done
done
done