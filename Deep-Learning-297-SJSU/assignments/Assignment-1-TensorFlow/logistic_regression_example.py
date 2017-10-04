import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

n_epochs = 10
batch_size = 128
learning_rate = 0.01

# Step 1: Read in data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True) 

# Step 2


# Step 3


# Step 4


# Step 5:


# Step 6:


with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs): 
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			# Run the optimizer and loss, store the loss value as loss_batch
			 
			 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))


	# test the model
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) 
	
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0
	
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}) 
		total_correct_preds += sum(accuracy_batch)
	
	print('Accuracy {0}'.format(accuracy_batch))
