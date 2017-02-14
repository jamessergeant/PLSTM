import numpy as np

from matplotlib import pyplot as plt

def get_batch(resolutions=None,batch_size=32,on=((74.,75.),(5., 6.)),off=((1., 100.),(1., 100.)),min_len=50,max_len=125,seq_len=1000):

    A_on_T = np.random.uniform(on[0][0],on[0][1],(batch_size,1))
    B_on_T = np.random.uniform(on[1][0],on[1][1],(batch_size,1))

    A_off_T1 = np.random.uniform(off[0][0],on[0][0],(batch_size,1))
    B_off_T1 = np.random.uniform(off[1][0],on[1][0],(batch_size,1))

    A_off_T2 = np.random.uniform(off[0][0],on[0][0],(batch_size,1))
    B_off_T2 = np.random.uniform(off[1][0],on[1][0],(batch_size,1))

    rA = np.random.uniform(0,1.,(batch_size,1)) > on[0][1] / off[0][1]
    rB = np.random.uniform(0,1.,(batch_size,1)) > on[1][1] / off[1][1]

    A_off_T = np.where(rA,A_off_T1,A_off_T2)
    B_off_T = np.where(rB,B_off_T1,B_off_T2)

    offsets = np.random.uniform(0., 100., (batch_size,1,))

    lengths = np.random.randint(seq_len/10,seq_len,(batch_size,))

    time_points_A = np.cumsum(np.asarray([resolutions[0]] * seq_len))
    time_points_B = np.cumsum(np.asarray([resolutions[1]] * seq_len))

    time_points = np.tile(np.unique([time_points_A,time_points_B])[:seq_len],[batch_size,1])

    time_points += np.tile(offsets,[1,time_points.shape[1]])

    A_on_T = np.tile(A_on_T,[1,time_points.shape[1]])
    B_on_T = np.tile(B_on_T,[1,time_points.shape[1]])
    A_off_T = np.tile(A_off_T,[1,time_points.shape[1]])
    B_off_T = np.tile(B_off_T,[1,time_points.shape[1]])

    A_on = np.sin(time_points * 2 * np.pi / A_on_T)
    B_on = np.sin(time_points * 2 * np.pi / B_on_T)

    A_off = np.sin(time_points * 2 * np.pi / A_off_T)
    B_off = np.sin(time_points * 2 * np.pi / B_off_T)

    A = A_on
    B = B_on
    B[int(batch_size/4):int(batch_size/2)] = B_off[int(batch_size/4):int(batch_size/2)]
    A[int(batch_size/2):int(3*batch_size/2)] = A_off[int(batch_size/2):int(3*batch_size/2)]
    A[int(3*batch_size/2):] = A_off[int(3*batch_size/2):]
    B[int(3*batch_size/2):] = B_off[int(3*batch_size/2):]

    y = np.zeros((batch_size,4))
    y[:int(batch_size/4),0] = 1.
    y[int(batch_size/4):int(batch_size/2),1] = 1.
    y[int(batch_size/2):int(3*batch_size/4),2] = 1.
    y[int(3*batch_size/4):,3] = 1.

    return time_points, A, B, y, lengths
