import tensorflow as tf

def cnn_model_fn(features, labels, mode):
	# Input
	input_layer = tf.reshape(features['image'], [-1, 384, 384, 3])

	# First thing's first, dump some images so I know that I didn't screw up my inputs
	tf.summary.image(
		name='input',
		tensor=input_layer,
	)

	# First convolutional layer
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu,
		name='conv1',
	)

	# Now we have [batch, 384, 384, 32] from the image size and the filter count

	# First pooling layer
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=2,
		strides=2,
		name='pool1',
	)

	# Now the shape is [batch, 192, 192, 32] by pooling by a factor of 2 in both dimensions

	# Second convolutional layer
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu,
		name='conv2',
	)

	# Up to [batch, 192, 192, 64] from the 64 filters

	# Second pooling layer
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2,
		pool_size=2,
		strides=2,
		name='pool2',
	)

	# Down to [batch, 96, 96, 64] from another 2x pooling factor

	# Flatten what we've got left
	pool2_flat = tf.reshape(pool2, [-1, 96 * 96 * 64])

	# Dense layer to actually put the pooled layers together
	dense = tf.layers.dense(
		inputs=pool2_flat,
		units=2048,
		activation=tf.nn.relu,
		name='dense',
	)

	# Dropout regularize the dense layer
	dropout = tf.layers.dropout(
		inputs=dense,
		rate=0.4,
		training=mode == tf.estimator.ModeKeys.TRAIN,
		name='dropout',
	)

	# Output (logits) layer
	logits = tf.layers.dense(inputs=dropout, units=1, name='logits', activation=None)

	predictions = {
		'score': logits,
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	# Is this loss?
	loss = tf.losses.mean_squared_error(
		labels=tf.reshape(labels, [-1, 1]),
		predictions=predictions['score'],
	)

	eval_metric_ops = {
		'mean_absolute_error': tf.metrics.mean_absolute_error(
			labels=labels,
			predictions=tf.cast(predictions['score'], dtype=tf.int64),
			name='mean_absolute_error',
		),
		#'global_step': [tf.train.get_global_step()],
		'rmse': tf.metrics.root_mean_squared_error(
			labels=labels,
			predictions=tf.cast(predictions['score'], dtype=tf.int64),
			name='rmse',
		),
		#'loss': loss,
	}

	# Choo choo motherfuckers
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-9)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step(),
		)
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
