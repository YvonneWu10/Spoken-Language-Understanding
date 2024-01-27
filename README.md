## 文件架构
```
├── data
├── word2vec-768.txt
├── HD_with_Bert             # all needed files for Hierarchical Decoder
├── MN                       # all needed files for MN
├── MN_Bert                  # all needed files for MN_Bert
├── MN_Transformer           # all needed files for MN_Transformer
├── MN_Bert_Transformer      # all needed files for MN_Bert_Transformer
├── SDEN                     # all needed files for SDEN
├── SDEN_Bert                # all needed files for SDEN_Bert
├── SDEN_Transformer         # all needed files for SDEN_Transformer
├── SDEN_Bert_Transformer    # all needed files for SDEN_Bert_Transformer
├── model                    # models used in baseline and CTran
├── scripts                  # scripts of the baseline model and CTran related models
│   ├── slu_baseline.py
│   ├── slu_CTran.py
│   └── slu_transformer.py 
├── utils                    # utils for baseline model and CTran       
├── predictions              # predictions of test_unlabelled.json
│   ├── asr_SDEN_Bert.json
│   ├── manual_SDEN_Transformer.json
└── ...
```


## Baseline
### 创建环境

    conda create -n slu python=3.11
    source activate slu
    pip install torch

### 运行训练脚本

在根目录下运行

    python scripts/slu_baseline.py

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成

      python scripts/slu_baseline.py --<arg> <value>

  其中，`<arg>`为要修改的参数名，`<value>`为修改后的值

+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU

+ `utils/vocab.py`:构建编码输入输出的词表

+ `utils/word2vec.py`:读取词向量

+ `utils/example.py`:读取数据

+ `utils/batch.py`:将数据以批为单位转化为输入

+ `model/slu_baseline_tagging.py`:baseline模型

+ `scripts/slu_baseline.py`:主程序脚本


## CTran

### 创建环境

    conda create -n slu python=3.11
    source activate slu
    pip install torch
    pip install einops==0.7.0
    pip install huggingface-hub==0.19.4
    pip install xpinyin==0.7.6
    pip install transformers==4.25.1

### 运行训练脚本

在根目录下运行

    python scripts/slu_baseline.py  
    python scripts/slu_CTran.py
    python scripts/slu_transformer.py


故在 `slu_baseline.py`与`slu_CTran.py` 有以下值得调整的参数：
- `batch_size`：默认32
- `lr`：学习率，默认1e-3
- `dropout`：默认为0.2
- `num_layer_rnn`：RNN深度，默认为2
- `trainset_spoken_language_select`：选择训练集，可选择['manual_transcript', 'asr_1best', 'both']，默认为'asr_1best'
- `encoder_cell`：RNN架构，可选择['LSTM', 'GRU', 'RNN']，默认为'LSTM'
- `word_embedding`：采用预训练模型或词向量，可选择['Word2vec', 'Bert','WWM', 'Roberta']，默认为'Word2vec'

故在 `slu_transformer.py` 有以下值得调整的参数：
- `batch_size`：默认32
- `lr`：学习率，默认1e-3
- `dropout`：默认为0.2
- `trainset_spoken_language_select`：选择训练集，可选择['manual_transcript', 'asr_1best', 'both']，默认为'asr_1best'
- `CNN`：是否使用CNN架构替代RNN
- `encoder_cell`：RNN架构，可选择['LSTM', 'GRU', 'RNN']，默认为'LSTM'
- `connection`：采用并行结构或叠加结构，可选择['Parallel', 'Serial']
- `num_layer_attn`：Transformer深度，默认为2
- `num_layer_rnn`：RNN深度，默认为2
- `word_embedding`：采用预训练模型或词向量，可选择['Word2vec', 'Bert','WWM', 'Roberta']，默认为'Word2vec'

### 运行测试脚本

在根目录下运行

    python scripts/slu_baseline.py --testing 
    python scripts/slu_CTran.py --testing 
    python scripts/slu_transformer.py --testing 

注意，此时需要先运行训练模型。测试结果将保存在根目录下`prediction.json`中。

## Hierarchical Decoder

### 创建环境

    conda create -n slu python=3.9
    source activate slu
    pip install torch==1.13.1
    pip instal einops==0.7.0
    pip install huggingface-hub==0.19.4
    pip install transformers==4.35.2

### 运行训练脚本

切换到目录./HD_with_Bert中运行以下代码
    
    python main.py


故在 `main.py`有以下值得调整的参数：
- `batch_size`：默认32
- `lr`：学习率，默认5e-4
- `encoder_cell`：RNN架构，可选择['LSTM', 'GRU', 'RNN']，默认为'RNN'
- `n_layers`：RNN深度，默认为3
- `dropout`：默认为0.2
- `noise`：bool类型变量，若为True则使用asr数据进行训练，若为False则使用manual transcript进行训练，默认为False
- `with_ptr`：bool类型变量，选择是否在value decoder中使用pointer network
- `dropout`：dropout rate at each non-recurrent layer，默认为0
- `bert_dropout`：dropout rate for BERT，默认为0.3
- `optim_choice`：训练优化器的选择，可从['schdadam', 'adam', 'adamw', 'bertadam']中选择，默认为bertadam
- `trans_layer`：encoder中transformer的层数，默认为3

### 运行测试脚本

切换到目录./HD_with_Bert中运行以下代码

    python main.py --testing 

注意，此时需要先运行训练模型。测试结果将保存在根目录下`./data/prediction.json`中。


## MN and SDEN

### 创建环境

    conda create -n slu python=3.8
    source activate slu
    pip install numpy==1.23.0
    pip install torch==2.1.1
    pip instal einops==0.7.0
    pip install huggingface-hub==0.20.1
    pip install transformers==4.36.1

### 运行训练脚本

切换到目录./MN or ./MN_Bert or ./MN_Transformer or ./MN_Bert_Transformer or ./SDEN or ./SDEN_Bert or ./SDEN_Transformer or ./SDEN_Bert_Transformer中运行以下代码
    
    python scripts/slu_baseline.py

在 `utils/args.py`中定义了涉及到的可选参数，有以下值得调整的参数：
- `lr`：学习率
- `c_encoder_cell`：current utterance的RNN架构，可选择['LSTM', 'GRU', 'RNN']
- `m_encoder_cell`：memory utterance的RNN架构，可选择['LSTM', 'GRU', 'RNN']
- `tagger_rnn`：tagger中的RNN架构，可选择['LSTM', 'GRU', 'RNN']
- `dropout`：tagger中的RNN dropout

如需改动某一参数可以在运行时将命令修改成

    python scripts/slu_baseline.py --<arg> <value>


### 运行测试脚本

切换到目录./MN or ./MN_Bert or ./MN_Transformer or ./MN_Bert_Transformer or ./SDEN or ./SDEN_Bert or ./SDEN_Transformer or ./SDEN_Bert_Transformer中运行以下代码

    python scripts/slu_baseline.py --testing

注意，此时需要先运行训练模型。测试结果将保存在根目录下`./data/prediction.json`中。