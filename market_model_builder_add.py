# from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, Add, merge, Conv2D,Flatten
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU


class MarketDeepQLearningModelBuilder():
   
    def __init__(self, action_size = 3):
        self.action_size = action_size
        self.learning_rate = 0.001

    def buildModel(self):

        input1 = Input(shape=(2,))
        input2 = Input(shape=(6,10,1))

        pre = []

        net1 = Dense(8, activation='relu')(input1)

        net2 = Conv2D(filters=1024, kernel_size=(1, 10), strides=1, data_format='channels_last')(input2)
        net2 = LeakyReLU(alpha = 0.001)(net2)
        net2 = Flatten()(net2)
        net2 = Dense(48)(net2)
        net2 = LeakyReLU(alpha = 0.001)(net2)
        
        pre.append(net1)
        pre.append(net2)

        hidden1 = concatenate(pre, axis=1)
        hidden2 = Dense(96, activation='relu')(hidden1)
        hidden3 = Dense(128, activation='relu')(hidden2)
        hidden4 = Dense(128, activation='relu')(hidden3)
        hidden5 = Dense(96, activation='relu')(hidden4)
        output_main = Dense(self.action_size, activation='linear')(hidden5)


        model = Model(inputs = [input1,input2], outputs = output_main)


        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        print (model.summary())

        return model



    # def buildModel(self):


        # merges = []
        # inputs = []

        # """B0"""
        # B0 = Input(shape=(2, ))
        # b = Dense(7, activation="sigmoid")(B0)

        # inputs.append(B0)
        # merges.append(b)

        # """S0"""
        # S0 = Input(shape=(7, 60, 1, ))

        # # s = Flatten()(S0)
        # s = Conv2D(filters=1024, kernel_size=(1, 60), strides=1, padding='valid', data_format='channels_last')(S0)
        # s = LeakyReLU(0.001)(s)
        # s = Flatten()(s)
        # s = Dense(90)(s)
        # s = LeakyReLU(0.001)(s)

        # inputs.append(S0)
        # merges.append(s)


        # m = concatenate(merges, axis=1)
        # m = Dense(90)(m)
        # m = LeakyReLU(0.001)(m)
        # m = Dense(30)(m)
        # m = LeakyReLU(0.001)(m)
                # V = Dense(3, activation='linear')(m)

        # model = Model(inputs=inputs, outputs=V)





