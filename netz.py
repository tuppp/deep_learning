import numpy as np
import pdb
import tensorflow as tf
from hyperparams import *

#hyperparams["filter"][1], hyperparams["filter"][0]
def ourConvolution(previous_layer, filter_height, filter_width, input_channels, output_channels):

    w = tf.Variable(tf.random_normal(shape=(filter_height, filter_width, input_channels, output_channels), mean=0,
                                     stddev=0.5))  # [filter_height, filter_width, in_channels, out_channels]

    layer1 = tf.nn.conv2d(
        input=previous_layer,
        filter=w,
        strides=[1, 1, 1, 1],
        padding="VALID",
        use_cudnn_on_gpu=True,
        data_format='NHWC',
        dilations=[1, 1, 1, 1],
        name=None
    )

    return layer1




trainX = np.array([
    [
        [10, 10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10, 10],
    ],
    [[10, 10, 10, 10, 10, 10, 10],
     [10, 10, 10, 10, 10, 10, 10],
     [10, 10, 10, 10, 10, 10, 10],
     [10, 10, 10, 10, 10, 10, 10],
     [10, 10, 10, 10, 10, 10, 10], ]

], dtype=float)

trainX = np.expand_dims(trainX, axis=3)
trainY = np.array([20, 20])


###Graphen###
hp = Hyperparams()
hyperparams = hp.getRandomHyperparameter()


y = tf.placeholder(tf.float32, shape=(None), name="test")
x = tf.placeholder(tf.float32, shape=(None, 5, 7, 1), name="test2")




init_op = tf.global_variables_initializer()



for i in hyperparams["nr_convs"]:
    if i==0:
        input_channel = 1
        output_channel = 10
        previous_layer = x
    else:
        previous_layer = layers[i-1]
        input_channel = hyperparams.channel_sequence[i-1]

    layers = []
    layers.append(ourConvolution(previous_layer,hyperparams["filter"][1], hyperparams["filter"][0], input_channel , hyperparams.channel_sequence[i]))

last_layer = layers[hyperparams["nr_convs"]-1]


pdb.set_trace()
# init_op = w.initializer


with tf.Session() as sess:
    sess.run(init_op)

    erg = sess.run([layer1], feed_dict={x: trainX, y: trainY})
    print erg
