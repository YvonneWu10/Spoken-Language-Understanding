lr_list=(5e-3 1e-3 1e-4)
dropout_list=(0.1 0.2 0.3 0.5)
num_layer_attn_list=(2 3 4)
word_embedding_list=('Word2vec' 'Bert' 'WWM' 'Roberta')

for lr in ${lr_list[@]};do
for dropout in ${dropout_list[@]};do
for num_layer_attn in ${num_layer_attn_list[@]};do
for word_embedding in ${word_embedding_list[@]};do
python scripts/slu_transformer.py --CNN --results_save \
        --connection Serial --word_embedding $word_embedding --lr $lr --dropout $dropout --num_layer_attn $num_layer_attn \
        --trainset_spoken_language_select manual_transcript  --device 6 --batch_size 256 

done
done
done
done
