lr_list=(5e-3 1e-3 1e-4)
dropout_list=(0.1 0.2 0.3 0.5)
num_layer_attn_list=(2 3 4)
encoder_cell_list=('LSTM' 'GRU')
word_embedding_list=('Word2vec' 'Bert' 'WWM' 'Roberta')
trainset_spoken_language_select_list=('manual_transcript' 'both')
connection_list=('Parallel' 'Serial')

for lr in ${lr_list[@]};do
for dropout in ${dropout_list[@]};do
for num_layer_attn in ${num_layer_attn_list[@]};do
for encoder_cell in ${encoder_cell_list[@]};do
for word_embedding in ${word_embedding_list[@]};do
for trainset_spoken_language_select in ${trainset_spoken_language_select_list[@]};do
for connection in ${connection_list[@]};do
python scripts/slu_transformer.py --lr $lr --dropout $dropout --num_layer_attn $num_layer_attn\
    --encoder_cell $encoder_cell --word_embedding $word_embedding --trainset_spoken_language_select $trainset_spoken_language_select \
    --connection $connection --device 3 --results_save --batch_size 512
done
done
done
done
done
done
done