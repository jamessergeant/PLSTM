from __future__ import division
import tensorflow as tf
import numpy as np
from tabulate import tabulate
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell
from PhasedLSTMCell import PhasedLSTMCell
from get_batch import get_batch

# Unit test for Phased LSTM
# Here I implement the first task described in the original paper of PLSTM
#   https://arxiv.org/abs/1610.09513
# which is the sine waves discrimination

flags = tf.flags
flags.DEFINE_string("unit", "MPLSTM", "Can be MPLSTM, PSLTM, LSTM, GRU")
flags.DEFINE_boolean("async", False, "Use asynchronous sampling")
flags.DEFINE_float("resolution", 0.1, "Sampling resolution if async is set to False")
flags.DEFINE_integer("n_hidden", 100, "hidden units in the recurrent layer")
flags.DEFINE_integer("n_epochs", 30, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("b_per_epoch", 80, "batches per epoch")
flags.DEFINE_integer("n_layers", 4, "hidden units in the recurrent layer")
flags.DEFINE_float("max_length", 125, "max length of sin waves")
flags.DEFINE_float("min_length", 50, "min length of sine waves")
flags.DEFINE_float("max_f_off", 100, "max frequency for the off set")
flags.DEFINE_float("min_f_off", 1, "min frequency for the off set")
flags.DEFINE_float("max_f_on", 5, "max frequency for the on set")
flags.DEFINE_float("min_f_on", 6, "min frequency for the on set")
flags.DEFINE_float("exp_init", 3., "Value for initialization of Tau")
FLAGS = flags.FLAGS

# Net Params
n_input = 1
n_out = 2
if FLAGS.async:
    tpe = "async"
else:
    tpe = "sync"
run_name = '{}_{}_res_{}_hid_{}_exp_{}'.format(FLAGS.unit, tpe, FLAGS.resolution, FLAGS.n_hidden, FLAGS.exp_init)

# Smart initialize for versions < 0.12.0
def initialize_all_variables(sess=None):
    """Initializes all uninitialized variables in correct order. Initializers
    are only run for uninitialized variables, so it's safe to run this multiple
    times.
    Args:
        sess: session to use. Use default session if None.
    """

    from tensorflow.contrib import graph_editor as ge
    def make_initializer(var):
        def f():
            return tf.assign(var, var.initial_value).op

        return f

    def make_noop():
        return tf.no_op()

    def make_safe_initializer(var):
        """Returns initializer op that only runs for uninitialized ops."""
        return tf.cond(tf.is_variable_initialized(var), make_noop,
                       make_initializer(var), name="safe_init_" + var.op.name).op

    if not sess:
        sess = tf.get_default_session()
    g = tf.get_default_graph()

    safe_initializers = {}
    for v in tf.all_variables():
        safe_initializers[v.op.name] = make_safe_initializer(v)

    # initializers access variable vaue through read-only value cached in
    # <varname>/read, so add control dependency to trigger safe_initializer
    # on read access
    for v in tf.all_variables():
        var_name = v.op.name
        var_cache = g.get_operation_by_name(var_name + "/read")
        ge.reroute.add_control_inputs(var_cache, [safe_initializers[var_name]])

    sess.run(tf.group(*safe_initializers.values()))

    # remove initializer dependencies to avoid slowing down future variable reads
    for v in tf.all_variables():
        var_name = v.op.name
        var_cache = g.get_operation_by_name(var_name + "/read")
        ge.reroute.remove_control_inputs(var_cache, [safe_initializers[var_name]])


def gen_async_sin(async_sampling, resolution=(0.1,0.15), batch_size=32, on_target_T=((74,75),(5, 6)), off_target_T=((1, 100),(1, 100)),
                  max_len=125, min_len=85):
    half_batch = int(batch_size / 2)
    quarter_batch = int(batch_size / 4)
    full_length_A = off_target_T[0][1] - on_target_T[0][1] + on_target_T[0][0] - off_target_T[0][0]
    # generate random periods
    posTs_A = np.random.uniform(on_target_T[0][0], on_target_T[0][1], half_batch)
    size_low_A = np.floor((on_target_T[0][0] - off_target_T[0][0]) * half_batch / full_length_A).astype('int32')
    size_high_A = np.ceil((off_target_T[0][1] - on_target_T[0][1]) * half_batch / full_length_A).astype('int32')
    low_vec_A = np.random.uniform(off_target_T[0][0], on_target_T[0][0], size_low_A)
    high_vec_A = np.random.uniform(on_target_T[0][1], off_target_T[0][1], size_high_A)
    negTs_A = np.hstack([low_vec_A,
                       high_vec_A])
    full_length_B = off_target_T[0][1] - on_target_T[0][1] + on_target_T[0][0] - off_target_T[0][0]
    # generate random periods
    posTs_B = np.random.uniform(on_target_T[1][0], on_target_T[1][1], half_batch)
    size_low_B = np.floor((on_target_T[1][0] - off_target_T[1][0]) * half_batch / full_length_B).astype('int32')
    size_high_B = np.ceil((off_target_T[1][1] - on_target_T[1][1]) * half_batch / full_length_B).astype('int32')
    low_vec_B = np.random.uniform(off_target_T[1][0], on_target_T[1][0], size_low_B)
    high_vec_B = np.random.uniform(on_target_T[1][1], off_target_T[1][1], size_high_B)
    negTs_B = np.hstack([low_vec_B,
                       high_vec_B])
    # generate random lengths
    if async_sampling:
        lens = np.random.uniform(min_len, max_len, batch_size)
    else:
        max_len_A = max_len * int(1 / resolution[0])
        min_len_A = min_len * int(1 / resolution[0])
        lens_A = np.random.uniform(min_len_A, max_len_A, batch_size)
        max_len_B = max_len * int(1 / resolution[1])
        min_len_B = min_len * int(1 / resolution[1])
        lens_B = np.random.uniform(min_len_B, max_len_B, batch_size)
    # generate random number of samples
    if async_sampling:
        samples_A = np.random.uniform(min_len_A, max_len_A, batch_size).astype('int32')
        samples_B = np.random.uniform(min_len_B, max_len_B, batch_size).astype('int32')
    else:
        samples_A = lens_A
        samples_B = lens_B
    samples = np.squeeze(np.zeros((1,batch_size)))
    start_times = np.array([np.random.uniform(0, max_len - duration) for duration in lens_A])
    xA = np.zeros((batch_size, max_len, 1))
    xB = np.zeros((batch_size, max_len, 1))
    y = np.zeros((batch_size, 4))
    t = np.zeros((batch_size, max_len, 1))
    for i, s, l, nA, nB in zip(range(batch_size), start_times, lens_A, samples_A, samples_B):
        if async_sampling:
            time_points_A = np.reshape(np.sort(np.random.uniform(s, s + l, nA)), [-1, 1])
            time_points_B = np.reshape(np.sort(np.random.uniform(s, s + l, nB)), [-1, 1])
        else:
            time_points_A = np.reshape(np.arange(s, s + nA * resolution[0], step=resolution), [-1, 1])
            time_points_B = np.reshape(np.arange(s, s + nB * resolution[1], step=resolution), [-1, 1])

        if i < quarter_batch:  # positive
            _tmp_xA = np.squeeze(np.sin(time_points_A * 2 * np.pi / posTs_A[i]))
            _tmp_xB = np.squeeze(np.sin(time_points_B * 2 * np.pi / posTs_B[i]))
            time_points = np.unique(np.sort(np.hstack([time_points_A.flatten(),time_points_B.flatten()])))
            in_A = np.in1d(time_points,time_points_A)
            in_A = np.pad(in_A,(0,xA.shape[1]-in_A.shape[0]),'constant',constant_values=0)
            in_B = np.in1d(time_points,time_points_B)
            in_B = np.pad(in_B,(0,xA.shape[1]-in_B.shape[0]),'constant',constant_values=0)
            t[i, :len(time_points), 0] = np.squeeze(time_points)
            xA[i, in_A, 0] = _tmp_xA
            xB[i, in_B, 0] = _tmp_xB
            y[i, 0] = 1.
        elif i >= quarter_batch and i < half_batch:
            _tmp_xA = np.squeeze(np.sin(time_points_A * 2 * np.pi / posTs_A[i]))
            _tmp_xB = np.squeeze(np.sin(time_points_B * 2 * np.pi / negTs_B[i - quarter_batch]))
            time_points = np.unique(np.sort(np.hstack([time_points_A.flatten(),time_points_B.flatten()])))
            in_A = np.in1d(time_points,time_points_A)
            in_A = np.pad(in_A,(0,xA.shape[1]-in_A.shape[0]),'constant',constant_values=0)
            in_B = np.in1d(time_points,time_points_B)
            in_B = np.pad(in_B,(0,xA.shape[1]-in_B.shape[0]),'constant',constant_values=0)
            t[i, :len(time_points), 0] = np.squeeze(time_points)
            xA[i, in_A, 0] = _tmp_xA
            xB[i, in_B, 0] = _tmp_xB
            y[i, 1] = 1.
        elif i >= half_batch and i < 3*quarter_batch:
            _tmp_xB = np.squeeze(np.sin(time_points_B * 2 * np.pi / posTs_B[i - quarter_batch]))
            _tmp_xA = np.squeeze(np.sin(time_points_A * 2 * np.pi / negTs_A[i - half_batch]))
            time_points = np.unique(np.sort(np.hstack([time_points_A.flatten(),time_points_B.flatten()])))
            in_A = np.in1d(time_points,time_points_A)
            in_A = np.pad(in_A,(0,xA.shape[1]-in_A.shape[0]),'constant',constant_values=0)
            in_B = np.in1d(time_points,time_points_B)
            in_B = np.pad(in_B,(0,xA.shape[1]-in_B.shape[0]),'constant',constant_values=0)
            t[i, :len(time_points), 0] = np.squeeze(time_points)
            xA[i, in_A, 0] = _tmp_xA
            xB[i, in_B, 0] = _tmp_xB
            y[i, 2] = 1.
        else:
            _tmp_xA = np.squeeze(np.sin(time_points_A * 2 * np.pi / negTs_A[i - half_batch]))
            _tmp_xB = np.squeeze(np.sin(time_points_B * 2 * np.pi / negTs_B[i - half_batch]))
            time_points = np.unique(np.sort(np.hstack([time_points_A.flatten(),time_points_B.flatten()])))
            in_A = np.in1d(time_points,time_points_A)
            in_A = np.pad(in_A,(0,xA.shape[1]-in_A.shape[0]),'constant',constant_values=0)
            in_B = np.in1d(time_points,time_points_B)
            in_B = np.pad(in_B,(0,xA.shape[1]-in_B.shape[0]),'constant',constant_values=0)
            t[i, :len(time_points), 0] = np.squeeze(time_points)
            xA[i, in_A, 0] = _tmp_xA
            xB[i, in_B, 0] = _tmp_xB
            y[i, 3] = 1.

        samples[i] = len(time_points)

    return t, xA, xB, y, samples, posTs_A, negTs_A

def multiPLSTM(mode1_input, mode2_input, times, mode1_period, mode2_period, batch_size, lens, n_layers, units_p_layer, n_input, initial_states,lstm_type='PLSTM'):
    """
    Function to build multilayer PLSTM
    :param input: 3D tensor, where the time input is appended and represents the last feature of the tensor
    :param batch_size: integer, batch size
    :param lens: 2D tensor, length of the sequences in the batch (for synamic rnn use)
    :param n_layers: integer, number of layers
    :param units_p_layer: integer, number of units per layer
    :param n_input: integer, number of features in the input (without time feature)
    :param initial_states: list of tuples of initial states
    :return: 3D tensor, output of the multilayer PLSTM
    """

    cells = []

    newX = tf.concat(2, [mode1_input,times])

    with tf.variable_scope("mode1"):
        cell = PhasedLSTMCell(units_p_layer, use_peepholes=True,
                                    state_is_tuple=True,manual_set=True, trainable=False,
                                    tau_init=1./mode1_period,r_on_init=1./(8*mode1_period))
        mode1_out, state = tf.nn.dynamic_rnn(cell, newX, dtype=tf.float32,
           sequence_length=lens,
           initial_state=initial_states[0])

    cells.append(cell)
    newX = tf.concat(2, [mode2_input,times])

    with tf.variable_scope("mode2"):
        cell = PhasedLSTMCell(units_p_layer, use_peepholes=True,
                                    state_is_tuple=True,manual_set=True, trainable=False,
                                    tau_init=1./mode_period,r_on_init=1./(8*mode1_period))
        mode2_out, state = tf.nn.dynamic_rnn(cell, newX, dtype=tf.float32,
                                                       sequence_length=lens,
                                                       initial_state=initial_states[1])
    cells.append(cell)
    newX = tf.concat(2, [mode1_out, mode2_out])

    for k in range(2,n_layers+2):
        newX = tf.concat(2, [newX, times])
        with tf.variable_scope("{}".format(k)):

            if lstm_type == 'PLSTM':
                cell = PhasedLSTMCell(units_p_layer, use_peepholes=True,
                                      state_is_tuple=True)
            elif lstm_type == 'MPLSTM':
                cell = ModPLSTMCell(units_p_layer, use_peepholes=True,
                            state_is_tuple=True)
            else:
                print('Invalid cell type')

            outputs, initial_states[k] = tf.nn.dynamic_rnn(cell, newX, dtype=tf.float32,
                                                           sequence_length=lens,
                                                           initial_state=initial_states[k])
            newX = outputs
            cells.append(cell)

    return newX

def RNN(_X, _weights, _biases, lens, initial_states, _periods):
    try:
        assert FLAGS.unit in ["MPLSTM","PLSTM","GRU","LSTM"]
    except:
        raise ValueError("Unit '{}' not implemented.".format(FLAGS.unit))

    outputs = multiPLSTM(_X[0], _X[1], _X[2], _periods[0], _periods[1], FLAGS.batch_size, lens, FLAGS.n_layers, FLAGS.n_hidden, n_input, initial_states, FLAGS.unit)

    outputs = tf.slice(outputs, [0, 0, 0], [-1, -1, FLAGS.n_hidden])

    # TODO better (?) in lack of smart indexing
    batch_size = tf.shape(outputs)[0]
    max_len = tf.shape(outputs)[1]
    out_size = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_len + (lens - 1)
    flat = tf.reshape(outputs, [-1, out_size])
    relevant = tf.gather(flat, index)

    return tf.nn.bias_add(tf.matmul(relevant, _weights['out']), _biases['out'])


def main(_):
    # inputs
    xA = tf.placeholder(tf.float32, [None, None, n_input + 1])
    xB = tf.placeholder(tf.float32, [None, None, n_input + 1])
    t = tf.placeholder(tf.float32, [None, None, 1])

    # length of the samples -> for dynamic_rnn
    lens = tf.placeholder(tf.int32, [None])

    # labels
    y = tf.placeholder(tf.float32, [None, 2])

    # weights from input to hidden
    weights = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden, n_out], dtype=tf.float32))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([n_out], dtype=tf.float32))
    }

    # Register weights to be monitored by tensorboard
    w_out_hist = tf.summary.histogram("weights_out", weights['out'])
    b_out_hist = tf.summary.histogram("biases_out", biases['out'])

    # Let's define the training and testing operations
    print ("Compiling RNN...",)
    initial_states = [None for _ in range(FLAGS.n_layers)]
    predictions = RNN(xA,xB, t, weights, biases, lens, initial_states)
    print ("DONE!")

    # Register initial_states to be monitored by tensorboard
    initial_states_hist = tf.summary.histogram("initial_states", initial_states[0][0])

    print ("Compiling cost functions...",)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y))
    print ("DONE!")

    # I like to log the gradients
    tvars = tf.trainable_variables()
    grads = tf.gradients(cost, tvars)

    grads_hist = [tf.summary.histogram("grads_{}".format(i), k) for i, k in enumerate(grads) if k is not None]
    merged_grads = tf.summary.merge([grads_hist] + [w_out_hist, b_out_hist] + [initial_states_hist])
    cost_summary = tf.summary.scalar("cost", cost)
    cost_val_summary = tf.summary.scalar("cost_val", cost)

    print ("Calculating gradients...",)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print ("DONE!")

    # evaluation
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    accuracy_val_summary = tf.summary.scalar("accuracy_val", accuracy)

    # run the model
    init = tf.global_variables_initializer()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.4)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        print ("Initializing variables...",)
        sess.run(init)
        # for backward compatibility (v < 0.12.0) use the following line instead of the above
        # initialize_all_variables(sess)
        print ("DONE!")

        writer = tf.summary.FileWriter("phasedLSTM_run/{}".format(run_name), sess.graph)

        # training loop
        for step in range(FLAGS.n_epochs):
            train_cost = 0
            train_acc = 0
            for i in range(FLAGS.b_per_epoch):
                batch_A_xs, batch_B_xs, batch_ys, leng, _, _ = gen_async_sin(FLAGS.async,
                                                                     FLAGS.resolution,
                                                                     FLAGS.batch_size, [FLAGS.min_f_on, FLAGS.max_f_on],
                                                                     [FLAGS.min_f_off, FLAGS.max_f_off],
                                                                     FLAGS.max_length,
                                                                     FLAGS.min_length)

                res = sess.run([optimizer, cost, accuracy, grads, cost_summary, accuracy_summary, merged_grads],
                               feed_dict={xA: batch_A_xs,
                                          xB: batch_B_xs,
                                          y: batch_ys,
                                          lens: leng
                                          })
                writer.add_summary(res[6], step * FLAGS.b_per_epoch + i)
                writer.add_summary(res[4], step * FLAGS.b_per_epoch + i)
                writer.add_summary(res[5], step * FLAGS.b_per_epoch + i)
                train_cost += res[1] / FLAGS.b_per_epoch
                train_acc += res[2] / FLAGS.b_per_epoch

            # test accuracy
            #wipe initial_states before testing
            for i, _ in enumerate(initial_states):
                initial_states[i] = None
            test_A_xs, test_B_xs, test_ys, leng, _, _ = gen_async_sin(FLAGS.async, FLAGS.resolution, FLAGS.batch_size,
                                                         [FLAGS.min_f_on, FLAGS.max_f_on],
                                                         [FLAGS.min_f_off, FLAGS.max_f_off],
                                                         FLAGS.max_length,
                                                         FLAGS.min_length)
            loss_test, acc_test, summ_cost, summ_acc = sess.run([cost,
                                                                 accuracy, cost_val_summary, accuracy_val_summary],
                                                                feed_dict={xA: test_A_xs,
                                                                           xB: test_B_xs,
                                                                           y: test_ys,
                                                                           lens: leng})
            writer.add_summary(summ_cost, step * FLAGS.b_per_epoch + i)
            writer.add_summary(summ_acc, step * FLAGS.b_per_epoch + i)
            table = [["Train", train_cost, train_acc],
                     ["Test", loss_test, acc_test]]
            headers = ["Epoch={}".format(step), "Cost", "Accuracy"]

            print (tabulate(table, headers, tablefmt='grid'))
            #wipe initial_states after testing
            for i, _ in enumerate(initial_states):
                initial_states[i] = None



if __name__ == "__main__":
    tf.app.run()
