python scripts/slu_transformer.py \
        --connection Serial --word_embedding Word2vec --CNN \
        --trainset_spoken_language_select manual_transcript  --device 6 --batch_size 256 


python scripts/slu_transformer.py --CNN --results_save \
        --connection Serial --word_embedding Bert --lr 5e-4 --dropout 0.3 --num_layer_attn 4 \
        --trainset_spoken_language_select manual_transcript  --device 5 --batch_size 512 --max_epoch 200

python scripts/slu_transformer.py \
        --connection Serial --word_embedding WWM --encoder_cell GRU --lr 1e-3 --dropout 0.5 --num_layer_attn 3 \
        --connection Serial --device 5 --batch_size 512 --max_epoch 200