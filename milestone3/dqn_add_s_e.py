import os
import numpy as np
import random
import copy
import csv
from env_add_s_e import MarketEnv
from model_builder_add import MarketDeepQLearningModelBuilder
from collections import deque
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from os import walk
from datetime import datetime
import matplotlib



class DeepQ:

    def __init__(self, env, gamma=0.85, model_file_name=None, test = False, random = False):
        self.env = env
        self.gamma = gamma
        self.model_filename = model_file_name
        self.memory = deque(maxlen=1000000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.005
        self.random = random

        self.model = MarketDeepQLearningModelBuilder().buildModel()
        self.fixed_model = MarketDeepQLearningModelBuilder().buildModel()


        if test:
            self.model.load_weights(self.model_filename)

    def epsilon_reset(self):
        self.epsilon = 1.0


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
            true_q.append(target_f)

            # checkpointer = ModelCheckpoint(filepath=self.model_filename, mode='auto', verbose=1, monitor='val_loss', save_best_only=True)
        train_states = np.array(train_states)
        true_q = np.array(true_q)
        history = self.model.fit(state, target_f, epochs=1, verbose=0)
        losses=history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return losses

    def act(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']


        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        
        if np.random.rand() <= self.epsilon:
            sample = self.env.action_space.sample()
            
            while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()

            return sample


        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]

        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1


        return act_index[flag] # returns action

    def act_predict_new(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']


        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        
        if np.random.rand() <= 0.1:
            sample = self.env.action_space.sample()
            
            while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()

            return sample


        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]

        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1


        return act_index[flag] # returns action


    def act_random(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)
        
        sample = self.env.action_space.sample()
        
        while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()
        return sample


    def act_predict(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']


        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        act_values = self.model.predict(state)[0]


        act_index = np.argsort(act_values)[::-1]
        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1

        return act_index[flag] # returns action



    def train(self, code_stocks, max_episode=100, verbose=0):

        history = open('./record/history_s.txt', 'a')

        for e in range(max_episode):
            self.env._reset(code_stocks)
            state = self.env._render()

            game_over = False
            reward_sum = 0

            holds = 0
            buys = 0
            sells = 0
            print ("----",self.env.targetCode,"----")
            info = {"trading_feeRate": self.env.trading_feeRate,"current_trading_price": self.env.current_trading_price, "current_asset_value":self.env.current_asset_value,\
                "current_asset":self.env.current_asset}
            while not game_over:

                action = self.act(state, info)

                next_state, reward, game_over, info = self.env._step(action)

                current_asset_value = info['current_asset_value']

                self.remember(state, action, reward, next_state, game_over)

                state = copy.deepcopy(next_state)


                if game_over:
                    toPrint = '----episode----', e,'totalgains:', round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3), 'mem size:', len(self.memory), '\n'
                    print (toPrint)
                    history.write(str(toPrint))
                    history.write('\n')
                    history.flush()

            self.model.save_weights(self.model_filename)
            print ('model weights saved')

            losses = self.replay(100)

            with open("./record/train_loss_s.txt","a") as file:
                file.write(str(losses)+'\n')

        history.close()

    def predict(self,code_stocks,return_1,acts):
        if os.path.exists('./record/test_history.txt'):
            os.remove('./record/test_history.txt')

        history = open('./record/test_history.txt', 'a')

        self.env._reset(code_stocks)
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


        print ("----",self.env.targetCode,"----")
        info = {"trading_feeRate": self.env.trading_feeRate,"current_trading_price": self.env.current_trading_price, "current_asset_value":self.env.current_asset_value,\
                "current_asset":self.env.current_asset}

        while not game_over:

            if self.random == True:
                action = self.act_random(state,info)
            else:
                action = self.act_predict(state,info)

            return_1.append(round((info['current_asset_value']-self.env.startAssetValue)/self.env.startAssetValue,3))
            acts.append(self.env.actions[action])
            next_state, reward, game_over, info = self.env._step(action)

            current_asset_value = info['current_asset_value']

            state = copy.deepcopy(next_state)


            if game_over:
                toPrint = '----episode----', methods,'totalgains:', round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3), 'mem size:', len(self.memory), '\n'
                print (toPrint)
                history.write(str(toPrint))
                history.write('\n')
                history.flush()

        history.close()
        return return_1, acts, round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3)

def exploreFolder(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f in filenames:
            files.append(f.replace(".csv", ""))
        break
    return files


if __name__ == "__main__":
    
    files = exploreFolder('./semantic_data/add')
    files.remove('.DS_Store')

    s_and_p_delete = ['QRVO', 'FOX', 'WLTW','ORCL', 'CFG', 'IQV', 'NAVI', 'DXC', 'NWS', 'FTV', 'KHC', 'ALLE', 'APTV', 'DWDP', 'PYPL', 'HPQ', 'WRK', 'HPE', 'CSRA', 'BHF', 'EVHC', 'FOXA', 'COTY', 'GOOG', 'UA', 'SYF', 'BHGE', 'INFO', 'NWSA', 'HLT']

    s_and_p = [x for x in files if x not in s_and_p_delete]

 
    s_and_p_test = sorted(s_and_p)


    if os.path.exists("./record/train_loss_s.txt"):
        os.remove("./record/train_loss_s.txt")

    if os.path.exists("./record/history_s.txt"):
        os.remove("./record/history_s.txt")
    
    env = MarketEnv(dir_path="./semantic_data/add/", target_codes=s_and_p, sudden_death_rate=0.3, finalIndex = 370) #370
    pg = DeepQ(env, gamma=0.80, model_file_name="./model/s_e.h5")
    
    for stock in s_and_p:
        pg.train(code_stocks = stock)

    
    env = MarketEnv(dir_path="./semantic_data/add/",target_codes=s_and_p_test, test = True, sudden_death_rate=0.3, finalIndex = 131) #131
    test_obj = DeepQ(env, gamma=0.80, model_file_name="./model/s_e.h5", test = True)
    
    reward_stock = []

    for stock in s_and_p_test:
        reward_collect = []
        return_1, acts, each_reward = test_obj.predict(stock,return_1,acts)
        reward_stock.append(each_reward)


    print(s_and_p_test)
    print(reward_stock)



