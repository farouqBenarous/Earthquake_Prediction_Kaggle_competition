import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import pandas as pd

# function for plotting based on both features
from tensorflow.python.training import saver


def plot_acc_ttf_data(Acousticdata, timeFailuer,
                      title="Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)
    plt.plot(Acousticdata, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()
    plt.plot(timeFailuer, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)
    del Acousticdata
    del timeFailuer

    plt.show()


if __name__ == "__main__":
    # ---------------------------- Bringing the data--------------------------------------
    pd.set_option("display.precision", 13)
    data = pd.read_csv("Data/train.csv",
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}, float_precision="hight")

    print(data.shape[0], data.shape[1])

    # Acousticdata = data.loc[:, ["acoustic_data"]].values
    Acousticdata = list(x for x in data["acoustic_data"])
    Acousticdata = np.array(Acousticdata).reshape(629145480, 1)
    # timeFailuer = data.loc[:, ["time_to_failure"]].values
    timeFailuer = list(x for x in data["time_to_failure"])
    timeFailuer = np.array(timeFailuer).reshape(629145480, 1)

    print(Acousticdata)
    # plot_acc_ttf_data(Acousticdata, timeFailuer, title="Acoustic data and time to failure: 1% sampled data")

    # -------------------------- Start buildin and training my model ----------------------

    display_step = 10  # to split the display

    # Create graph
    tf.reset_default_graph()
    sess = tf.Session()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))

    # make results reproducible
    seed = 13
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Declare batch size
    batch_size = 50

    # Initialize placeholders
    x_data = tf.placeholder(shape=(629145480, None), dtype=tf.float32)
    y_target = tf.placeholder(shape=(629145480, None), dtype=tf.float32)

    # Create variables for linear regression
    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # Declare model operations
    model_output = tf.add(tf.matmul(x_data, A), b)

    # Declare the elastic net loss function
    elastic_param1 = tf.constant(1.)
    elastic_param2 = tf.constant(1.)

    l1_a_loss = tf.reduce_mean(tf.abs(A))
    l2_a_loss = tf.reduce_mean(tf.square(A))

    e1_term = tf.multiply(elastic_param1, l1_a_loss)
    e2_term = tf.multiply(elastic_param2, l2_a_loss)

    # define the loss function and the optimazer
    loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # Initialize variabls and tensorflow session
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    # Training loop
    loss_vec = []
    with sess:
        for i in range(100):
            sess.run(optimizer, feed_dict={x_data: Acousticdata, y_target: timeFailuer})
            if (i) % display_step == 0:
                temp_loss = sess.run(loss, feed_dict={x_data: Acousticdata, y_target: timeFailuer})
                loss_vec.append(temp_loss[0])
                print("Training step:", '%04d' % (i), "cost=", temp_loss)
                save_path = saver.save(sess, "D:\Earthquake_Prediction_Kaggle_competition/models/model.ckpt")

        # answer = tf.equal(tf.floor(model_output + 0.1), y_target)
        # accuracy = tf.reduce_mean(tf.cast(answer, "float32"))
        # print(accuracy.eval(feed_dict={x_data: Acousticdata, y_target: timeFailuer}, session=sess))

        # answer = sess.run(model_output, feed_dict={x_data:12})
        # print( answer)»:_;

    # loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)
