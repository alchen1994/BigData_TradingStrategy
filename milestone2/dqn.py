import os
import numpy as np
import random
import copy
from env import MarketEnv
from model_builder import MarketDeepQLearningModelBuilder
from collections import deque
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from os import walk



class DeepQ:

    def __init__(self, env, gamma=0.85, model_file_name=None, test = False, random = False):
        self.env = env
        self.gamma = gamma
        self.model_filename = model_file_name
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.  
        self.epsilon_decay = .98
        self.epsilon_min = 0.005
        self.random = random

        self.model = MarketDeepQLearningModelBuilder().buildModel()
        self.fixed_model = MarketDeepQLearningModelBuilder().buildModel()


        if test:
            self.model.load_weights(self.model_filename)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):

        batchs = min(batch_size, len(self.memory))
        batchs = np.random.choice(len(self.memory), batchs)
        losses = []

        true_q = []
        train_states = []
        self.fixed_model.set_weights(self.model.get_weights())
        for i in batchs:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                predict_f = self.fixed_model.predict(next_state)[0]

                target = reward + self.gamma * np.amax(predict_f)

            target_f = self.model.predict(state)

            target_f[0][action] = target
            

            train_states.append(state[0])
            true_q.append(target_f[0])

            # checkpointer = ModelCheckpoint(filepath=self.model_filename, mode='auto', verbose=1, monitor='val_loss', save_best_only=True)
        train_states = np.array(train_states)
        true_q = np.array(true_q)
        history = self.model.fit(train_states, true_q, epochs=1, verbose=0)
        losses=history.history['loss'][0]
        return losses


    def act(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        cur_baseAmount = info["baseAmount"]


        trading_fee_general = float(cur_baseAmount * info["current_trading_price"])
        trading_fee_buy = trading_fee_general*(1+info["trading_feeRate"])
        trading_fee_sell = info["trading_feeRate"]*trading_fee_general
        
        if np.random.rand() <= self.epsilon:
            sample = self.env.action_space.sample()
            
            while (cur_cash < trading_fee_buy and sample == 0) or \
            (cur_stock < cur_baseAmount and sample == 1) or (cur_cash < trading_fee_sell and sample == 1):
                sample = self.env.action_space.sample()

            return sample


        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]

        flag = 0

        while (cur_cash < trading_fee_buy and act_index[flag] == 0) or \
            (cur_stock < cur_baseAmount and act_index[flag]== 1) or \
            (cur_cash < trading_fee_sell and act_index[flag] == 1):
            flag+=1


        return act_index[flag] 

    def act_random(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        cur_baseAmount = info["baseAmount"]

        trading_fee_general = float(cur_baseAmount * info["current_trading_price"])
        trading_fee_buy = trading_fee_general*(1+info["trading_feeRate"])
        trading_fee_sell = info["trading_feeRate"]*trading_fee_general
        sample = self.env.action_space.sample()
        
        while (cur_cash < trading_fee_buy and sample == 0) or \
        (cur_stock < cur_baseAmount and sample == 1) or (cur_cash < trading_fee_sell and sample == 1):
            sample = self.env.action_space.sample()
        return sample


    def act_predict(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        cur_baseAmount = info["baseAmount"]


        trading_fee_general = float(cur_baseAmount * info["current_trading_price"])
        trading_fee_buy = trading_fee_general*(1+info["trading_feeRate"])
        trading_fee_sell = info["trading_feeRate"]*trading_fee_general

        act_values = self.model.predict(state)[0]


        act_index = np.argsort(act_values)[::-1]
        flag = 0

        while (cur_cash < trading_fee_buy and act_index[flag] == 0) or \
            (cur_stock < cur_baseAmount and act_index[flag]== 1) or \
            (cur_cash < trading_fee_sell and act_index[flag] == 1):
            flag+=1


        return act_index[flag] 


    def train(self, max_episode=900, verbose=0):

        history = open('./record/history.txt', 'w')

        if os.path.exists("./record/train_loss.txt"):
            os.remove("./record/train_loss.txt")

        for e in range(max_episode):
            self.env._reset()
            state = self.env._render()

            game_over = False
            reward_sum = 0

            holds = 0
            buys = 0
            sells = 0
            print ("----",self.env.code,"----")
            info = { "baseAmount": 2,"trading_feeRate": 0.001,"current_trading_price": 0, \
                "current_asset":{'cash':200, 'stock':0}}
            while not game_over:

                action = self.act(state, info)

                next_state, reward, game_over, info = self.env._step(action)
                
                current_asset_value = info['current_asset_value']

                self.remember(state, action, reward, next_state, game_over)

                state = copy.deepcopy(next_state)

                if game_over:
                    toPrint = '----episode----', e,'totalgains:', round((current_asset_value-200)/200,3), 'mem size:', len(self.memory), '\n'
                    print (toPrint)
                    history.write(str(toPrint))
                    history.write('\n')
                    history.flush()

            if e % 20 == 0 and e != 0:
                self.model.save_weights(self.model_filename)
                print ('model weights saved')

            losses = self.replay(128)

            with open("./record/train_loss.txt","a") as file:
                file.write(str(losses)+'\n')

        history.close()

    def predict(self):
        if os.path.exists('./record/test_history.txt'):
            os.remove('./record/test_history.txt')

        history = open('./record/test_history.txt', 'a')

        self.env._reset()
        state = self.env._render()

        game_over = False
        reward_sum = 0

        holds = 0
        buys = 0
        sells = 0

        if self.random == True:
            methods = 'random'
        else:
            methods = 'DQN'


        print ("----",self.env.code,"----")
        info = { "baseAmount": 2,"trading_feeRate": 0.001,"current_trading_price": 0, \
                "current_asset":{'cash':200, 'stock':0}}

        while not game_over:

            if self.random == True:
                action = self.act_random(state,info)
            else:
                action = self.act_predict(state,info)

            next_state, reward, game_over, info = self.env._step(action)
            
            current_asset_value = info['current_asset_value']

            state = copy.deepcopy(next_state)

            if game_over:
                toPrint = '----episode----', methods,'totalgains:', round((current_asset_value-200)/200,3), 'mem size:', len(self.memory), '\n'
                print (toPrint)
                history.write(str(toPrint))
                history.write('\n')
                history.flush()

        history.close()
        return round((current_asset_value-200)/200,3)

def exploreFolder(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f in filenames:
            files.append(f.replace(".csv", ""))
        break
    return files


if __name__ == "__main__":

    s_and_p = ['ADI', 'AJG', 'APD', 'CVX', 'DLR', 'DVA', 'ETN', 
    'HES', 'INTU', 'IT','L', 'MAR', 'MET', 'MMM', 'NOC', 'NSC', 'PLD', 'SPGI', 'TJX', 'TMO']

    for stock in s_and_p:
        env = MarketEnv(dir_path="./split_data/train/", target_codes=stock, sudden_death_rate=0.3, finalIndex = 997) #1259
        pg = DeepQ(env, gamma=0.80, model_file_name="./model/model_"+stock+".h5")
        pg.train()

    reward_stock = []
    reward_stock_random = []

    for stock in s_and_p:
        env = MarketEnv(dir_path="./split_data/test/", target_codes=stock, sudden_death_rate=0.3, finalIndex = 256)
        test_obj = DeepQ(env, gamma=0.80, model_file_name="./model/model_"+stock+".h5", test = True)
        test_obj_random = DeepQ(env, gamma=0.80, model_file_name="./model/model_"+stock+".h5", test = True, random = True)


        reward_collect = []
        reward_collect_random = []

        for i in range(20):
            each_reward = test_obj.predict() 
            reward_collect.append(each_reward)
            each_reward = test_obj_random.predict()
            reward_collect_random.append(each_reward)

        reward_stock.append(np.mean(reward_collect))
        reward_stock_random.append(np.mean(reward_collect_random))

    plt.plot(s_and_p,reward_stock)
    plt.plot(s_and_p,reward_stock_random)
    plt.title("DQN vs Random")
    plt.xlabel("stock")
    plt.ylabel("return")
    plt.legend(['DQN', 'Random'], loc='upper left')
    plt.show()




