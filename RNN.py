import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.rnn.ptb import reader

# 数据存放的路径
DATA_PATH = "./path/to/ptb/data"
# 隐藏层的规模
HIDDEN_SIZE = 200

NUM_LAYERS = 2  # 深层神经网络中的LSTM的结构层数
VOCAB_SIZE = 10000  # 词典的规模,加上句尾的标识符和稀有单词标识符一共一万个单词.

LEARNING_RATE = 1.0  # 设置学习的速率
TRAIN_BATCH_SIZE = 20  #训练时候每一个batch的大小
TRAIN_NUM_STEP = 35  # 训练数据的截断数据

EVAL_BATCH_SIZE = 1  # 测试数据的batch大小
EVAL_NUM_STEP = 1  # 测试数据的截断长度
NUM_EPOCH = 2  # 使用训练数据的轮数
KEEP_PROB = 0.5  # 节点不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的参数


# 首先创建一个PTBModel来描述模型,这样方便维护循环神经网络中的状态
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch的大小和截断的长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层, 可以看到输入层的维度为batch_size* num_steps,这里
        # 和ptb_iterator函数的输出的训练数据batch是一致的.
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义预期输出, 他的维度和ptb_iterator函数的输出也是一样的.
        self.targets = tf.placeholder(tf.float32, [batch_size, num_steps])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper\
                (lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS)

        # 初始化最初的状态,也就是全零的向量.
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 将单词的ID转化为单词向量,因为一共有VOCAB_SIZE个单词,每个单词的向量维度为
        # HIDDEN_SIZE,所以embedding参数的维度为VOCAB_SIZE*HIDDEN_SIZE
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本batch_size * num_steps个单词ID转换为单词向量,转化后的输入层维度
        # 为batch_size * num_steps * HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        # 只在训练时候使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)
        # 定义输出列表.在这里香江不同时刻的LSTM结构的输出收集起来,然后通过一个全连接
        # 层得到最终的输出.
        outputs = []
        # state存储不同的batch中的LSTM的状态,并将其初始化为0
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step,:], state)
                outputs.append(cell_output)

        # 把队列展开成[batch, hidden_size*num_steps]的形状,然后在reshape成
        # [batch*num_steps, hidden_size]的形状.
        output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_SIZE])



