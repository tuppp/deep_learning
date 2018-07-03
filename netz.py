#np.ones((5,7))
#city(zeile), zeit(spalte)

#batchsize,city,zeile




#7*5

import numpy as np
import pdb
import tensorflow as tf
from hyperparams import *
import csv

def getDataSequence(batchsize,city1,city2,city3,city4,city5):
    with open('data.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        l5 = []
        for row in spamreader:
            if row[0]==city1:
                l1.append(row)
            if row[0]==city2:
                l2.append(row)
            if row[0]==city3:
                l3.append(row)
            if row[0]==city4:
                l4.append(row)
            if row[0]==city5:
                l5.append(row)
        #print(len(l1),len(l2),len(l3),len(l4),len(l5),)
        ok=0
        if l1[0][3]==l2[0][3]:
            ok+=1
        if l1[0][3]==l3[0][3]:
            ok+=1
        if l1[0][3]==l4[0][3]:
            ok+=1
        if l1[0][3]==l5[0][3]:
            ok+=1

        res=[]
        if ok==4:
            #print("Data ist OK")
            if batchsize*8>len(l1):
                #print("Die Batches sind zu gross")
                return None
            else:
                batchamount=len(l1)//(batchsize*8)
                print(batchamount)
                for i in range(batchamount):
                    batch=np.zeros((batchsize,5,7))
                    batchres=np.zeros((batchsize))
                    for j in range(batchsize):
                        tmp = np.array(
                            [
                                [item[15] for item in l1[i*8*batchsize+j*8:i*8*batchsize++j*8+7]],
                                [item[15] for item in l2[i*8*batchsize+j*8:i*8*batchsize++j*8+7]],
                                [item[15] for item in l3[i*8*batchsize+j*8:i*8*batchsize++j*8+7]],
                                [item[15] for item in l4[i*8*batchsize+j*8:i*8*batchsize++j*8+7]],
                                [item[15] for item in l5[i*8*batchsize+j*8:i*8*batchsize++j*8+7]],
                            ])
                        batch[j]=tmp
                        batchres[j]=l1[i*8*batchsize+j*8+7][15]
                    res.append([batch,batchres])

            #print(len(res))
            return res
        else:
            #print("Data ist fehlerhaft")
            return None






batches=getDataSequence(5   ,"1504","102","3319","4560","2638")
for batch in batches:
    trainX=batch[0]
    trainY=batch[1]
    #print(trainX)
    #print(trainY)




#readAllData




def fc_layer(input, neuronsize):
    shape=(input.shape[1].value, neuronsize)
    w = tf.Variable(tf.random_normal(shape=shape, stddev=0.03), name='w')
    b = tf.Variable(tf.zeros(neuronsize), name="biases")
    layer = tf.matmul(input, w)
    layer_biased = tf.add(layer,b)
    return layer_biased


#hyperparams["filter"][1], hyperparams["filter"][0]
def ourConvolution(previous_layer, filter_height, filter_width, input_channels, output_channels):
    w = tf.Variable(tf.random_normal(shape=(filter_height, filter_width, input_channels, output_channels), mean=0,
                                     stddev=0.5))  # [filter_height, filter_width, in_channels, out_channels]
    b = tf.Variable(tf.zeros(output_channels), name="biases")
    layer1 = tf.nn.conv2d(
        input=previous_layer,
        filter=w,
        strides=[1, 1, 1, 1],
        padding="SAME",
        use_cudnn_on_gpu=True,
        data_format='NHWC',
        dilations=[1, 1, 1, 1],
        name=None
    )

    layer1_biased = tf.add(layer1,b)

    return layer1_biased

def ourPooling(conv_layer, kheight, kwidth):
    ksize = [1, kheight, kwidth, 1]
    pool_layer = tf.nn.max_pool(
        value=conv_layer,
        ksize=ksize,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format='NHWC',
        name=None
    )

    return pool_layer




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
keep_prob = tf.placeholder(tf.float32)






#convolution layers
layers = []
for i in np.arange(hyperparams["nr_convs"]):
    if i==0:
        input_channel = 1
        output_channel = 10
        previous_layer = x
    else:
        previous_layer = layers[(i*2)-1]
        input_channel = hyperparams["channel_sequence"][i-1]

    conv_layer = ourConvolution(previous_layer,hyperparams["filter"][1], hyperparams["filter"][0], input_channel , hyperparams["channel_sequence"][i])
    layers.append(conv_layer)
    layers.append(ourPooling(conv_layer,hyperparams["ksize"][1], hyperparams["ksize"][0]))

last_conv_layer = layers[hyperparams["nr_convs"]*2-1]


concat_layer = tf.reshape(last_conv_layer, [-1, last_conv_layer.shape[1].value*last_conv_layer.shape[2].value*last_conv_layer.shape[3].value] )



#fully connected layers
fc_layers = []
for i in range(hyperparams["nr_fully_connected_layers"]):
    print i

    if i ==0:
        previous_layer = concat_layer
    else:
        previous_layer = fc_layers[i-1]



    iteration_fc_layer = fc_layer( previous_layer, hyperparams["nr_neurons_in_convlayer"])
    fc_layers.append(iteration_fc_layer)

last_fc_layer = fc_layers[hyperparams["nr_fully_connected_layers"]-1]



#dense layer
dense_layer = fc_layer(last_fc_layer,1)
loss = tf.losses.mean_squared_error(labels = y, predictions=dense_layer)
optimiser = tf.train.AdamOptimizer(learning_rate=hyperparams["learning_rate"]).minimize(loss)


init_op = tf.global_variables_initializer()

with tf.Session() as sess:




    n = 1000
    batch_size = hyperparams["batch_size"]
    batches_per_fold = hyperparams["batches_per_fold"]
    current_pointer = 0
    loss_array = []
    accuracy_array = []

    sess.run(init_op)

    for fold in range(0, 5):
        getData.val_id = i
        getData.fold_shuffle()
        testdata = getData.test
        testdata_labels = getData.test_labels
        valdata = getData.val
        valdata_labels = getData.val_labels
        for epoch in range(0, n):
            size = testdata.shape[2]
            indices = np.arange(size)
            np.random.shuffle(indices)
            batch = np.ones(5,7,size)
            batch_val = np.ones(1,1,size)
            for batches in range(0, batches_per_fold*4):
                for i in range(current_pointer, batch_size):
                    batch[:,:,i] = testdata[:,:,indices[i]]
                    batch_val[i] = valdata[indices[i]]
                current_pointer = current_pointer + batch_size
                loss_o, _ = sess.run([loss, optimiser], feed_dict={batch: trainX, batch_val: trainY})
                loss_array.append(loss_o)
