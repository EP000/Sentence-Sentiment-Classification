import pandas as pd
import numpy as np
train = pd.read_csv("C:/Users/PC/Desktop/AIR/data/movie review/train.tsv", delimiter='\t', header=0)
test_x = pd.read_csv("C:/Users/PC/Desktop/AIR/data/movie review/test.tsv", delimiter='\t', header=0)
test_y = pd.read_csv("C:/Users/PC/Desktop/AIR/data/movie review/Submission.csv", delimiter=',', header=0)
test = pd.merge(test_x,test_y)

#단어
words = train[train['Phrase'].str.split().map(len) == 1]
#words.index

# 문장들만 추출하기
def sub_sentence(input_df):
    return input_df.index[0]
sentences = train.iloc[train.groupby('SentenceId')['PhraseId'].agg([sub_sentence])['sub_sentence']]

# 문장의 길이 분포
#temp = pd.DataFrame({'len': sentences['Phrase'].str.split().map(len)})
#temp.groupby('len')['len'].agg('count')

# 가장 긴 문장 보기
#temp = train[train['Phrase'].str.split().map(len) == 52]

#train.shape
#test.shape

#temp = train[train['SentenceId']==17]
 
#단어사전         
word_dict = {word[3]: [word[1], word[4]] for word in words.itertuples()}
sentence_length = np.array(train['Phrase'].str.split().map(len))
max_sentence_length = sentence_length.max()

#단어-감정 태깅
x_train = []
for sent in sentences['Phrase']:
    sent_words = [word_dict[sent_word] if sent_word in word_dict else [-1, -1] for sent_word in sent.split()]
    sent_words.extend([[0, 0] for _ in range(max_sentence_length - len(sent_words))])
    x_train.append(sent_words)

#모델링
import tensorflow as tf
tf.reset_default_graph()

x_train = np.array(x_train)
y_train = np.array(sentences['Sentiment'])
y_train = np.eye(5)[y_train]

#X_one_hot = tf.one_hot(x_train[])

nb_classes = 1
#Y_one_hot = tf.one_hot(y_train, nb_classes)
#Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

learning_rate = 0.001
total_epoch = 30
batch_size = 128

n_input = 1
n_step = max_sentence_length
n_hidden = 128
n_class = 5

X = tf.placeholder(tf.float32,[None,n_step,n_input])
Y = tf.placeholder(tf.float32,[None,n_class])
seq_length = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden,n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seq_length)

model = tf.matmul(states, W) + b
#model = tf.contrib.layers.fully_connected(states,n_class,activation_fn=None)
#model = tf.layers.dense(states,n_class,activation=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    total_batch = int(x_train.shape[0]/batch_size)
    
    for epoch in range(total_epoch):
        total_cost = 0
        
        for i in range(total_batch):
            batch_xs = x_train[i*batch_size:(i+1)*batch_size]
            batch_ys = y_train[i*batch_size:(i+1)*batch_size]
            batch_length = sentence_length[i*batch_size:(i+1)*batch_size]
            
            _, cost_val = sess.run([optimizer, cost],feed_dict={X:batch_xs[:, :, 1:2], Y:batch_ys, 
                                   seq_length: batch_length})
            total_cost += cost_val
        
        print('Epoch :', '%04d' % (epoch+1), 'Avg. cost = ','{:3f}'.format(total_cost/total_batch))
    print('Optimization Complete')
    
    is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))