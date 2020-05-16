# Word2VecSeq2SeqEncdoerDecoder
第三章 问答摘要与推理-Seq2Seq（一）

## 一、Homework-week3
### 1. 构建Seq2seq模型中的Encoder层，采用面向对象的编程形式:
```python
class Encoder():
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        # code
    def call(self, x, hidden):
        # code
        pass

    def initialize_hidden_state(self):
        pass
```

### 2. 构建Seq2seq模型中的Decoder层，采用面向对象的编程形式:
```python
class Decoder():
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Decoder, self).__init__()
        # code
    def call(self, x, hidden, enc_output, context_vector):
        # code
        pass
```

### 3. 构建Seq2seq模型中的Attention层，采用面向对象的编程形式:
```python
class BahdanauAttention():
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # code

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        # code

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        # code

        # attention_weights shape == (batch_size, max_length, 1)
        # code

        # context_vector shape after sum == (batch_size, hidden_size)
        # code

        return 
```

### 4. 结合前两节课完成代码，完成整个项目训练部分代码，达到跑通训练的基本要求。


## 二、功能实现
1、原始数据AutoMaster_TestSet.csv和AutoMaster_TrainSet.csv都只保留50条，便于快速调试跑通模型  
2、在encoder.py中实现encoder层  
3、在decoder.py中实现decoder层  
4、 在attention.py中实现attention层    
5、在seq2seq.py中实现seq2seq模型  
6、在losses.py中定义loss函数  
