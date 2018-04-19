from random import random
import numpy as np
import math

import gym
from gym import spaces


class MarketEnv(gym.Env):
    # PENALTY = 0.9987

    def __init__(self, dir_path, target_codes, sudden_death_rate=0.5, test = False, startAssetValue = 5000, current_asset = {'cash':5000, 'stock':0}, finalIndex = 1997):

        self.sudden_death_rate = sudden_death_rate
        self.current_asset = current_asset
        # self.startAssetValue = current_asset['cash']
        self.target_codes = target_codes
        self.dataCollect = {}
        self.test = test
        self.finalIndex = finalIndex


        self.startAssetValue = startAssetValue 
        # self.baseAmount = baseAmount
        self.current_trading_price = 0
        self.last_trading_price = 0
        self.current_asset_value = 0
        self.last_asset_value = 0
        self.trading_feeRate = 0.0001
        self.range = 10

        for code in target_codes:
            data = []
            # data_new = []
            # data_temp =[]

            fn = dir_path + code + ".csv"

            # lastDate = 0
            lastOpen = 0
            lastHigh = 0
            lastLow = 0
            lastClose = 0
            lastVolume = 0
            lastPrice = 0
            lastscore = 0

            #0.041243687 102.61  105.368 102 105.35  67649387
            # print(fn)
            f = open(fn, "r")
            for line in f:
                if line.strip() != "":
                    score, openPrice, high, low, close, volume = line.strip().split(",")
                    # print(dt)
#
                    score = float(score)
                    openPrice = float(openPrice) if openPrice != "" else float(close)
                    high = float(high) if high != "" else float(close)
                    low = float(low) if low != "" else float(close)
                    close = float(close)
                    volume = float(volume)
                    trading_price = (low+high+close)/3
                    

                    if lastClose > 0 and close > 0 and lastVolume > 0:
                        # date_ = (date - lastDate)
                        open_ = (openPrice - lastOpen)/lastOpen
                        high_ = (high - lastHigh)/lastHigh
                        low_ = (low - lastLow)/lastLow
                        close_ = (close - lastClose)/lastClose
                        volume_ = (volume - lastVolume)/lastVolume
                        score_ = score
                        # trading_price_ = (trading_price-lastPrice)

                        
                        # data[dt]=[date_, open_, high_, low_, close_, trading_price_, volume_]
                        data.append([open_, high_, low_, close_, volume_, trading_price, score_, score])
                        # self.data.append([date_, open_, high_, low_, close_, volume_])

                    # lastDate = date
                    lastOpen = openPrice
                    lastHigh = high
                    lastLow = low
                    lastClose = close
                    lastVolume = volume
                    lastPrice = trading_price

            f.close()
            
            # data_temp = data.copy()
            # data_temp_0 = np.array(data_temp)

            # data_temp_1 = data_temp_0[:,:-1]
            # data_temp_2 = data_temp_0[:,:-1].mean(0)

            # data_temp_3 = data_temp_1/data_temp_2

            # Xnew = np.hstack((data_temp_3,data_temp_0[:,-1].reshape(len(data_temp),1)))
            # data_new = Xnew.tolist()
            # print(data_new[0])
            # exit()
            self.dataCollect[code] = data


        self.actions = [
            "Buy",
            "Sell",
            "Hold",
        ]

        self.action_space = spaces.Discrete(len(self.actions))


        # self._reset(code_stocks)


    # def convertDate(self, date):
    #     date = date.split('-')
    #     date = date[1] + '.' + date[2]
    #     date = float(date)
    #     return date


    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.last_trading_price = self.stock_data[self.currentIndex][5]


        self.last_asset_value = round(self.current_asset['stock'] * self.last_trading_price + self.current_asset['cash'], 3)
        
 
        price_change = self.stock_data[self.currentIndex+2][5] -  self.last_trading_price

        score = self.stock_data[self.currentIndex][-1]
        score_change = self.stock_data[self.currentIndex+1][-1] - score

        if self.actions[action] == "Buy":
            cur_stock = self.current_asset['stock']
            cur_cash = self.current_asset['cash']
            trading_amount = int(cur_cash/(self.last_trading_price*(1+self.trading_feeRate)))

            self.current_asset['stock'] = trading_amount +cur_stock
            self.current_asset['cash'] = 0.0

            # self.reward = float((self.current_asset_value - self.last_asset_value)/float(self.last_asset_value))
            self.defineState()

            self.currentIndex += 1
            self.current_trading_price = self.stock_data[self.currentIndex][5]

            self.current_asset_value = round((self.current_asset['stock']* self.current_trading_price), 3)
            
            # self.reward = self.current_asset_value - self.last_asset_value + price_change * trading_amount
            self.reward = (self.current_asset_value - self.last_asset_value) + price_change * trading_amount- self.current_asset['stock']+score_change*trading_amount
            #10
            # self.reward = (self.current_asset_value - self.last_asset_value)


        elif self.actions[action] == "Sell":
            cur_cash = self.current_asset['cash']
            cur_stock = self.current_asset['stock']
            trading_amount = cur_stock
            
            self.current_asset['cash'] = cur_cash+trading_amount * self.last_trading_price*(1-self.trading_feeRate)
            self.current_asset['stock'] = 0

            self.defineState()
            self.currentIndex += 1
            self.current_trading_price = self.stock_data[self.currentIndex][5]

            self.current_asset_value = round(self.current_asset['stock']* self.current_trading_price+ self.current_asset['cash'], 3)

            # self.reward = (self.current_asset_value - self.last_asset_value)


            self.reward = (self.current_asset_value - self.last_asset_value) - price_change* trading_amount + self.current_asset['stock']-score_change*trading_amount
            #7


        elif self.actions[action] == "Hold":
            self.defineState()
            self.currentIndex += 1

            self.reward = 0
            # -20
        else:
            pass

        # self.defineState()

        # self.currentIndex += 1

        # self.current_trading_price = float(self.data[self.currentIndex][5])


        # self.current_asset_value = round(((float(self.current_asset['stock']) * self.current_trading_price) + float(self.current_asset['cash'])), 3)

        if self.currentIndex >= self.finalIndex \
                or self.current_asset_value < self.sudden_death_rate * self.startAssetValue:
                 # or self.convertDate(self.endDate)<= self.data[self.currentIndex][0]:
            self.done = True

        # self.reward = float((self.current_asset_value - self.last_asset_value)/float(self.last_asset_value))



        return self.state, self.reward, self.done, \
               {"trading_feeRate": self.trading_feeRate,\
                "current_trading_price": self.current_trading_price, \
                "current_asset":self.current_asset, "cur_reward": self.reward,
                "act":self.actions[action], "last_asset_value":self.last_asset_value, "current_asset_value":self.current_asset_value}


    def _reset(self,code_stocks):
        self.current_asset = {'cash':self.startAssetValue, 'stock':0}
        #randomly choose a targetCode
        # self.targetCode = self.target_codes[int(random() * len(self.target_codes))]
        self.targetCode= code_stocks

        self.stock_data = self.dataCollect[self.targetCode]

        # self.targetDates = sorted(self.target.keys())
        if self.test == True:
            self.currentIndex = 380#380
        self.currentIndex = 10

        self.done = False
        self.reward = 0

        self.current_trading_price = 0
        self.last_trading_price = 0

        self.current_asset_value = self.startAssetValue
        self.last_asset_value = self.startAssetValue
        # self.range = 10

        self.defineState()


        return self.state


    def _render(self):
        return self.state


    # def _seed(self):
    #     return int(random() * 100)



    def defineState(self):
        tmpState = []

        self.current_trading_price = self.stock_data[self.currentIndex][5]
        # self.startAssetValue
        cashFeature = self.current_asset['cash']/self.startAssetValue
        stockFeature = self.current_asset['stock'] * self.current_trading_price/self.startAssetValue
        
        tmpState.append([[cashFeature, stockFeature]])
        
        openpart = []
        highpart = []
        lowpart = []
        closepart = []
        volumnpart = []
        scorepart = []


        # open_, high_, low_, close_, volume_, trading_price

        for i in range(self.range):
            openpart.append([self.stock_data[self.currentIndex-1-i][0]])
            highpart.append([self.stock_data[self.currentIndex-1-i][1]])
            lowpart.append([self.stock_data[self.currentIndex-1-i][2]])
            closepart.append([self.stock_data[self.currentIndex-1-i][3]])
            volumnpart.append([self.stock_data[self.currentIndex-1-i][4]])
            scorepart.append([self.stock_data[self.currentIndex-1-i][6]])

        tmpState.append([[openpart, highpart, lowpart,closepart, volumnpart, scorepart]])
        # tmpState = self.stock_data[self.currentIndex][:-1].copy()
        tmpState = [np.array(i) for i in tmpState]
        # tmpState.append(cashFeature)
        # tmpState.append(stockFeature)
        # tmpState1 = np.array(tmpState[0])
        # tmpState1.reshape([1, tmpState1.shape[0]])
        # # print(tmpState1.shape)
        # tmpState2 = np.array(tmpState[1])
        # tmpState2.reshape([1, tmpState2.shape[0], tmpState2.shape[1]])
        # # print(tmpState2.shape)
        # # exit()
        # # tmpState = np.array(tmpState)
        # tmpState_new.append(tmpState1)
        # tmpState_new.append(tmpState2)


        # print(tmpState[0].shape)
        # print(tmpState[1].shape)
        # exit()
        # print  (tmpState.shape)
        # exit()
        # tmpState = tmpState.reshape([1, tmpState.shape[0]])
        # print(tmpState.shape)
        # exit()
        self.state = tmpState




        # self.state = self.data[self.currentIndex]
        # self.state = np.array(self.state)
        # self.state = self.state.reshape([1, self.state.shape[0]])

        # print(self.state)
        # exit()


        # tmpState.append([[cashProfile, stockProfile]])






        # subjectDate = []
        # subjectHigh = []
        # subjectLow = []
        # subjectClose = []
        # subjectVolume = []
        # subjectTradingPrice = []
        # subjectOpen = []

        # for i in range(self.scope):
        #     # try:
        #         subjectDate.append([self.target[self.targetDates[self.currentIndex - 1 - i]][0]])  # date
        #         subjectHigh.append([self.target[self.targetDates[self.currentIndex - 1 - i]][1]]) #high
        #         subjectLow.append([self.target[self.targetDates[self.currentIndex - 1 - i]][2]]) #low
        #         subjectClose.append([self.target[self.targetDates[self.currentIndex - 1 - i]][3]]) #close
        #         subjectVolume.append([self.target[self.targetDates[self.currentIndex - 1 - i]][4]]) #volume
        #         subjectTradingPrice.append([self.target[self.targetDates[self.currentIndex - 1 - i]][5]])  # trading price
        #         subjectOpen.append([self.target[self.targetDates[self.currentIndex - 1 - i]][6]])  # open

            # except Exception, e:
                # print ('error in market_env.defineState')
                # print (self.targetCode, self.currentIndex, i, len(self.targetDates))
                # self.done = True

        # tmpState.append([[subjectDate, subjectHigh, subjectLow, subjectClose, subjectVolume, subjectTradingPrice, subjectOpen]])

        # tmpState = [np.array(i) for i in tmpState]

        # self.state = tmpState

    def printState(self):
        # print (self.state)
        print (self.state)


        # if self.current_asset['cash'] > 0:
        #     self.current_asset['cash'] = self.current_asset['cash'] - (10+ random() * 3)

        # current_asset_value = float(self.current_asset['stock']) * current_trading_price+ float(self.current_asset['cash'])





            # self.current_asset['cash'] = round( (float(self.current_asset['stock']) * float(current_trading_price) + cur_cash) , 3) \
            # - self.trading_feeRate * float(self.current_asset['stock']*float(current_trading_price))
            # self.current_asset['stock'] = round(float(0), 3)



            # self.current_asset['stock'] = round((float(self.current_asset['cash']) / float(current_trading_price) + cur_stock), 3)
            # self.current_asset['cash'] = round(float(0), 3) - self.trading_feeRate * float(self.current_asset['cash'])


        # except Exception, e:
            # print (e)

        # if len(data.keys()) > scope:
        #     self.dataMap[code] = data
        #     if code in target_codes:
        #         self.targetCodes.append(code)

        # 
        # self.observation_space = spaces.Box(np.ones(scope * (1)) * -1,np.ones(scope * (1)))

    # def convertDate(self, date):
    #     print(date)
    #     date = date.split('/')
    #     date = date[1] + '.' + date[0]
    #     date = float(date)
    #     return date