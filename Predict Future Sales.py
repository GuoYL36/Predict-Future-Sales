import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('./data/sales_train_v2.csv')
df_test = pd.read_csv('./data/test.csv')
df_items = pd.read_csv('./data/items.csv')
df_submit = pd.read_csv('./data/sample_submission.csv')

# df_train.head()
# df_test.head()
# df_items.head()
# df_submit.head()
#print(df_items.head())
#print(str(df_items.columns.values[2])==' item_id')
item_dict = df_items[['item_id','item_category_id']].to_dict()
#item_dict
df_train['item_cat_id'] = pd.Series()

df_train['item_cat_id'] = df_train['item_id'].apply(lambda x:item_dict['item_category_id'][x])
# df_train.head()

# len(df_train)
# len(df_test)
# print('每天的出售数量最小值：',df_train['item_cnt_day'].unique().min())
# print('每天的出售数量最大值：',df_train['item_cnt_day'].unique().max())

# items = ['shop_id','item_cat_id','date_block_num']
#
# for item in items:
#     plt.figure(figsize=(25,25))
#     item_counts = df_train[item].value_counts()
#     sns.barplot(item_counts.index,item_counts.values)
#     plt.title(item+' count')
#     plt.show()


# df_train[df_train['item_price']<0].count()
# df_train[df_train['item_cnt_day']<0].count()
# df_train[df_train['item_price']<0]
# df_train[df_train['item_cnt_day']<0]

df_train=df_train[(df_train['item_price']>0)&(df_train['item_cnt_day']>0)]
# len(df_train)

dataset=df_train.pivot_table(index=['item_id','shop_id'],columns=['date_block_num'],values=['item_cnt_day'],fill_value=0)
# dataset.head()
# print(dataset.head())
# print(df_test.head())
dataset_filtered = pd.merge(df_test, dataset, on=['item_id','shop_id'],how='left')
dataset_filtered.fillna(0, inplace=True)

dataset1 = dataset_filtered.drop(['ID','shop_id','item_id'],axis=1).values

# dataset1[:5]
# 制作数据集，取(item_cnt_day,0)~(item_cnt_day,32)的数据来预测(item_cnt_day,33)
# 并从以上数据集中分割训练和验证集
data_target = dataset1[:,33]
data_train = dataset1[:,0:33]
# data_train.shape

data_train = np.expand_dims(data_train, axis=2)
# data_train.shape

def batch_generator(batch):
    size = data_train.shape[0]
    array0 = np.arange(size)
    np.random.shuffle(array0)
    train_data = data_train[array0,:,:]
    train_label = data_target[array0]
    i = 0
    while True:
        if i+batch <= size:
            yield train_data[i:i+batch,:,:], train_label[i:i+batch]
            i += batch
        else:
            i = 0
            array0 = np.arange(size)
            np.random.shuffle(array0)
            train_data = data_train[array0,:,:]
            train_label = data_target[array0]
            continue
dims = 1
n_steps = 33
batch_size = 64
UNITS = [8,16]

with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
    ema_op = variable_averages.apply(tf.trainable_variables())
    x = tf.placeholder(shape=[None, n_steps, dims],
                       dtype=tf.float32,
                       name='x')  # [batch_size, n_steps, dims] = [64, 33, 1]
    print('x: ', x)
    y_ = tf.placeholder(shape=[None, ], dtype=tf.float32, name='y_')
    print('y_: ', y_)
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
    lstm_cell_0 = tf.nn.rnn_cell.LSTMCell(num_units=UNITS[0], name="lstm_cell_0")  # num_units相应于输入的dims
    lstm_cell_0 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_0, output_keep_prob=keep_prob)
    print('lstm_cell_0: ', lstm_cell_0)
    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=UNITS[1], name="lstm_cell_1")
    print('lstm_cell_1: ', lstm_cell_1)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_0, lstm_cell_1])
    print('lstm_cell: ', lstm_cell)
    output, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
    print('final_state: ', final_state)  # c和h是最后一个step的隐层状态，维度为[batch_size, num_units] = [64, 16]
    print('output: ', output)  # [batch_size, n_steps, num_units] = [64, 33, 16]
with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
    output = tf.reshape(output, [-1, n_steps * UNITS[1]])
    w = tf.Variable(tf.truncated_normal([n_steps * UNITS[1], 1], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[1]))
    out = tf.nn.xw_plus_b(output, w, b)     # [batch_size, 1]
    out = tf.squeeze(out)
    print('out: ', out)
    outputs = tf.nn.relu(out)
with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
    loss = tf.reduce_mean(tf.square(y_ - outputs))
    # loss = tf.losses.mean_squared_error(y_,outputs)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss, global_step=global_step)
    train_op = tf.group([optimizer, ema_op])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# store ckpt model
saver = tf.train.Saver(max_to_keep=5) #

EPOCH = 10
for epoch in range(EPOCH):
    losses = 0
    for i in range(3347):
        train, label = next(batch_generator(batch_size))
        feed_dict0 = {x: train, y_: label, keep_prob: 0.8}
        _, loss0 = sess.run([train_op, loss], feed_dict=feed_dict0)
        losses += loss0
        if i % 1000 == 0:
            print("loss0: ", loss0)
    losses /= 3347  # calculate mean loss
    print('epoch: %d, mean loss: %.2f' % (epoch, losses))
steps = sess.run(global_step)
saver.save(sess, './ckpt/model.ckpt', global_step=steps)

# ------Predict
test_data = dataset1[:,1:34]
test_data = np.expand_dims(test_data, axis=2)
out0 = sess.run(out,feed_dict={x:test_data,keep_prob:1.0})
out0 = np.clip(out0,0,20)
np.max(out0), np.min(out0)

#------write to csv file
submission = pd.DataFrame({'ID': df_test['ID'], 'item_cnt_month': out0})
submission.to_csv('submission.csv',index=False)

