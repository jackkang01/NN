import xlrd
import numpy as np
import tensorflow as tf

data = xlrd.open_workbook('ML_data1.xlsx') 
table = data.sheet_by_name('Sheet2') 
table1 = data.sheet_by_name('Sheet3')


data1 = np.zeros((452, 279), dtype=float) 
data1 = np.float32(data1)
label = np.zeros((452, 16), dtype=float)
label = np.float32(label)

for i in range(452):
	for j in range(279):
		if(np.max(table.col_values(j)) == np.min(table.col_values(j)) ==0):
			data1[i, j] = 0
		else:
			data1[i, j] = (table.cell(i, j).value - np.min(table.col_values(j))) / (np.max(table.col_values(j)) - np.min(table.col_values(j)))
			#data1[i, j] = table.cell(i, j).value
for a in range(452):
	for b in range(1, 17):
		if(table1.cell(a+1, 0).value == b):
			label[a, b-1] = 1

train_data = data1[:300, :]
train_label = label[:300, :]
test_data = data1[301:451, :]
test_label = label[301:451, :]


x = tf.placeholder("float", [None, 279])
y = tf.placeholder("float", [None, 16])
def add_layer(inputs, input_node, output_node, active_function=None):
	w1 = tf.Variable(tf.random_normal([input_node, output_node], stddev=0.1))
	b1 = tf.Variable(tf.random_normal([output_node]))
	y0 = tf.add(tf.matmul(inputs, w1), b1)
	#y0 = tf.nn.dropout(y0, 0.8)
	if(active_function==None):
		outputs = y0
	else:
		outputs = active_function(y0)
	return outputs

y1 = add_layer(x, input_node=279, output_node=128, active_function=tf.nn.tanh)
y2 = add_layer(y1, input_node=128, output_node=64, active_function=tf.nn.tanh)
y3 = add_layer(y2, input_node=64, output_node=16)

loss = tf.reduce_mean(tf.square(y3 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
corr = tf.equal(tf.argmax(y3, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

init = tf.global_variables_initializer()
print("FUNCTION READY")
saver = tf.train.Saver()
sess= tf.Session()
work_mode = "train"         
if(work_mode == "train"):
	sess.run(init)
	for i in range(1,20000):
		sess.run(train_step, feed_dict={x: train_data, y: train_label})
		train_accr = sess.run(accr, feed_dict={x: train_data, y: train_label})
		
		if(i%500==0):
			print("train-accr: %.3f" % train_accr)
			test_accr = sess.run(accr, feed_dict={x: test_data, y:test_label})
			if(test_accr > 0.68):         
				saver.save(sess, "Model/model.ckpt")
				print("good_test: %.3f" % test_accr)

if(work_mode == "test"):
	saver.restore(sess, "./Model/model.ckpt")
	TestAccr = sess.run(accr, feed_dict={x: test_data, y: test_label})
	print("accr: %.3f" % a)
