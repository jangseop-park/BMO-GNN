from spektral.layers import GCNConv, GCSConv, GlobalAvgPool, MinCutPool
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Add, Lambda, Dropout
from keras.initializers import RandomNormal
from keras import backend as K

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def gcs_mincut(N, F, n_out, params, name):  # library version
    batch_size = 1
    H_in = Input(shape=(N, F), batch_size=batch_size)
    A_in = Input(shape=(N, N), batch_size=batch_size)

    X_1 = GCSConv(channels=params['dim'], activation='relu')([H_in, A_in])
    X_pool_1, A_pool_1 = MinCutPool(N)([X_1, A_in])
    X_2 = GCSConv(channels=params['dim'], activation='relu')([X_pool_1, A_pool_1])
    X_pool_2, A_pool_2 = MinCutPool(N)([X_2, A_pool_1])
    X_3 = GCSConv(channels=params['dim'], activation='relu')([X_pool_2, A_pool_2])

    out = GlobalAvgPool()(X_3)

    out = dense(params['dense1'])(out)
    out = dense(params['dense2'])(out)
    out = dense(params['dense3'])(out)
    out = dense(n_out, activation='linear')(out)


    # out = Dense(params['dense1'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # # out = Dropout(0.5)(out)
    # out = Dense(params['dense2'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # # out = Dropout(0.5)(out)
    # out = Dense(params['dense3'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # out = Dense(n_out)(out)

    return Model(inputs=[H_in, A_in], outputs=out, name=name)



def gcs3(N, F, n_out, params, name):  # library version
    batch_size = 1
    H_in = Input(shape=(N, F), batch_size=batch_size)
    A_in = Input(shape=(N, N), batch_size=batch_size)

    # X_1 = GCSConv(channels=params['dim'], activation='relu')([H_in, A_in])
    # X_pool_1, A_pool_1 = MinCutPool(500)([X_1, A_in])
    # X_2 = GCSConv(channels=params['dim'], activation='relu')([X_pool_1, A_pool_1])
    # X_pool_2, A_pool_2 = MinCutPool(250)([X_2, A_pool_1])
    # X_3 = GCSConv(channels=params['dim'], activation='relu')([X_pool_2, A_pool_2])

    X_1 = GCSConv(channels=params['dim'], activation='relu')([H_in, A_in])
    X_2 = GCSConv(channels=params['dim'], activation='relu')([X_1, A_in])
    X_3 = GCSConv(channels=params['dim'], activation='relu')([X_2, A_in])
    out = GlobalAvgPool()(X_3)

    out = dense(params['dense1'])(out)
    out = dense(params['dense2'])(out)
    out = dense(params['dense3'])(out)
    out = dense(n_out, activation='linear')(out)


    # out = Dense(params['dense1'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # # out = Dropout(0.5)(out)
    # out = Dense(params['dense2'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # # out = Dropout(0.5)(out)
    # out = Dense(params['dense3'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # out = Dense(n_out)(out)

    return Model(inputs=[H_in, A_in], outputs=out, name=name)

def gcn3(N, F, n_out, params):  # library version
    batch_size = 1
    H_in = Input(shape=(N, F), batch_size=batch_size)
    A_in = Input(shape=(N, N), batch_size=batch_size)

    X_1 = GCNConv(channels=params['dim'], activation='relu')([H_in, A_in])
    # X_1 = Dropout(0.5)(X_1)
    X_2 = GCNConv(channels=params['dim'], activation='relu')([X_1, A_in])
    # X_2 = Dropout(0.5)(X_2)
    X_3 = GCNConv(channels=params['dim'], activation='relu')([X_2, A_in])
    # X_3 = Dropout(0.5)(X_3)
    out = GlobalAvgPool()(X_3)

    out = dense(params['dense1'])(out)
    out = dense(params['dense2'])(out)
    out = dense(params['dense3'])(out)
    out = dense(n_out)(out)


    # out = Dense(params['dense1'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # # out = Dropout(0.5)(out)
    # out = Dense(params['dense2'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # # out = Dropout(0.5)(out)
    # out = Dense(params['dense3'], dtype='float64')(out)
    # # out = BatchNormalization(dtype='float64')(out)
    # out = Activation('relu', dtype='float64')(out)
    # out = Dense(n_out)(out)

    return Model(inputs=[H_in, A_in], outputs=out)


def gcn4(N, F, n_out, params):  # library version
    batch_size = 1
    H_in = Input(shape=(N, F), batch_size=batch_size)
    A_in = Input(shape=(N, N), batch_size=batch_size)

    X_1 = GCNConv(channels=params['dim'], activation='relu')([H_in, A_in])
    X_2 = GCNConv(channels=params['dim'], activation='relu')([X_1, A_in])
    X_3 = GCNConv(channels=params['dim'], activation='relu')([X_2, A_in])
    out = GlobalAvgPool()(X_3)


    out = Dense(params['dense1'], dtype='float64')(out)
    # out = BatchNormalization(dtype='float64')(out)
    out = Activation('relu', dtype='float64')(out)
    out = Dense(params['dense2'], dtype='float64')(out)
    # out = BatchNormalization(dtype='float64')(out)
    out = Activation('relu', dtype='float64')(out)
    out = Dense(params['dense3'], dtype='float64')(out)
    # out = BatchNormalization(dtype='float64')(out)
    out = Activation('relu', dtype='float64')(out)
    out = Dense(n_out)(out)

    return Model(inputs=[H_in, A_in], outputs=out)

def dense(n_hidden, activation='relu',
          init_stddev=0.1, init_mean=0.0, data_type='float64',
          seed=None):
    kernel_initializer = RandomNormal(mean=init_mean, stddev=init_stddev, seed=seed)
    bias_initializer = RandomNormal(mean=init_mean, stddev=init_stddev, seed=seed)
    return Dense(n_hidden, activation=activation,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 dtype=data_type,
                 use_bias=True)

def matmul(XY):

    X,Y = XY
    return K.tf.matmul(X,Y)


def GAP(X):
    return K.tf.reduce_mean(X, axis=1, keepdims=True)


def gcn_custom(N, F, n_out, params):

    batch_size = 1
    H_in = Input(shape=(N, F), batch_size=batch_size)
    A_in = Input(shape=(N, N), batch_size=batch_size)

    h1 = dense(params['dim'])(Lambda(matmul)([A_in, H_in]))
    h2 = dense(params['dim'])(Lambda(matmul)([A_in, h1]))
    h3 = dense(params['dim'])(Lambda(matmul)([A_in, h2]))
    gap = Lambda(GAP)(h3)
    gap = Lambda(lambda y: K.squeeze(y, 1))(gap)
    out = dense(params['dense1'])(gap)
    # out = Dropout(0.5)(out)
    out = dense(params['dense2'])(out)
    # out = Dropout(0.5)(out)
    out = dense(params['dense3'])(out)
    out = dense(n_out)(out)

    return Model(inputs=[H_in, A_in], outputs=out)



def gcn15(N, F, n_out, params):  # library version
    batch_size = 1
    H_in = Input(shape=(N, F), batch_size=batch_size)
    A_in = Input(shape=(N, N), batch_size=batch_size)

    X_512 = Dense(params['dim'])(H_in)
    X_512 = BatchNormalization()(X_512)
    X_512 = Activation('relu')(X_512)
    X_512 = Dense(params['dim'])(X_512)
    X_512 = BatchNormalization()(X_512)
    X_512 = Activation('relu')(X_512)

    graph_conv_1 = GCNConv(channels=params['dim'], activation='relu')([X_512, A_in])
    X_1 = Add()([X_512, graph_conv_1])
    graph_conv_2 = GCNConv(channels=params['dim'], activation='relu')([X_1, A_in])
    X_2 = Add()([X_1, graph_conv_2])
    graph_conv_3 = GCNConv(channels=params['dim'], activation='relu')([X_2, A_in])
    X_3 = Add()([X_2, graph_conv_3])
    graph_conv_4 = GCNConv(channels=params['dim'], activation='relu')([X_3, A_in])
    X_4 = Add()([X_3, graph_conv_4])
    graph_conv_5 = GCNConv(channels=params['dim'], activation='relu')([X_4, A_in])
    X_5 = Add()([X_4, graph_conv_5])
    graph_conv_6 = GCNConv(channels=params['dim'], activation='relu')([X_5, A_in])
    X_6 = Add()([X_5, graph_conv_6])
    graph_conv_7 = GCNConv(channels=params['dim'], activation='relu')([X_6, A_in])
    X_7 = Add()([X_6, graph_conv_7])
    graph_conv_8 = GCNConv(channels=params['dim'], activation='relu')([X_7, A_in])
    X_8 = Add()([X_7, graph_conv_8])
    graph_conv_9 = GCNConv(channels=params['dim'], activation='relu')([X_8, A_in])
    X_9 = Add()([X_8, graph_conv_9])
    graph_conv_10 = GCNConv(channels=params['dim'], activation='relu')([X_9, A_in])
    X_10 = Add()([X_9, graph_conv_10])
    graph_conv_11 = GCNConv(channels=params['dim'], activation='relu')([X_10, A_in])
    X_11 = Add()([X_10, graph_conv_11])
    graph_conv_12 = GCNConv(channels=params['dim'], activation='relu')([X_11, A_in])
    X_12 = Add()([X_11, graph_conv_12])
    graph_conv_13 = GCNConv(channels=params['dim'], activation='relu')([X_12, A_in])
    X_13 = Add()([X_12, graph_conv_13])
    graph_conv_14 = GCNConv(channels=params['dim'], activation='relu')([X_13, A_in])
    X_14 = Add()([X_13, graph_conv_14])
    graph_conv_15 = GCNConv(channels=params['dim'], activation='relu')([X_14, A_in])
    X_15 = Add()([X_14, graph_conv_15])

    out = GlobalAvgPool()(X_15)
    out = Dense(params['dense1'])(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(params['dense2'])(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(params['dense3'])(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(n_out, 'linear')(out)

    return Model(inputs=[H_in, A_in], outputs=out, name="test")
