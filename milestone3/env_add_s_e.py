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
        self.target_codes = target_codes
        self.dataCollect = {}
        self.test = test
        self.finalIndex = finalIndex


        self.startAssetValue = startAssetValue 
        self.current_trading_price = 0
        self.last_trading_price = 0
        self.current_asset_value = 0
        self.last_asset_value = 0
        self.trading_feeRate = 0.0001
        self.range = 10

        for code in target_codes:
            data = []

            fn = dir_path + code + ".csv"

            lastOpen = 0
            lastHigh = 0
            lastLow = 0
            lastClose = 0
            lastVolume = 0
            lastPrice = 0
            lastscore = 0

            f = open(fn, "r")
            for line in f:
                if line.strip() != "":
                    score, openPrice, high, low, close, volume = line.strip().split(",")


                    score = float(score)
                    openPrice = float(openPrice) if openPrice != "" else float(close)
                    high = float(high) if high != "" else float(close)
                    low = float(low) if low != "" else float(close)
                    close = float(close)
                    volume = float(volume)
                    trading_price = (low+high+close)/3
                    

                    if lastClose > 0 and close > 0 and lastVolume > 0 and lastscore!=0:

                        open_ = (openPrice - lastOpen)/lastOpen
                        high_ = (high - lastHigh)/lastHigh
                        low_ = (low - lastLow)/lastLow
                        close_ = (close - lastClose)/lastClose
                        volume_ = (volume - lastVolume)/lastVolume
                        score_ = (score-lastscore)/lastscore

                        
                        data.append([open_, high_, low_, close_, volume_, trading_price, score_,score])


                    lastOpen = openPrice
                    lastHigh = high
                    lastLow = low
                    lastClose = close
                    lastVolume = volume
                    lastPrice = trading_price
                    lastscore = score

            f.close()

            self.dataCollect[code] = data


        self.actions = [
            "Buy",
            "Sell",
            "Hold",
        ]

        self.action_space = spaces.Discrete(len(self.actions))



    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.current_trading_price = self.stock_data[self.currentIndex][5]

 
        price_change = self.stock_data[self.currentIndex+1][5] - self.current_trading_price
        
        score = self.stock_data[self.currentIndex][-1]
        score_change = self.stock_data[self.currentIndex+1][-1] - score

        if self.actions[action] == "Buy":
            cur_stock = self.current_asset['stock']
            cur_cash = self.current_asset['cash']
            trading_amount = int(cur_cash/(self.current_trading_price*(1+self.trading_feeRate)))

            self.current_asset['stock'] = trading_amount + cur_stock
            self.current_asset['cash'] = cur_cash - trading_amount*self.current_trading_price*(1+self.trading_feeRate) 

            self.reward = (price_change+score_change)*trading_amount


        elif self.actions[action] == "Sell":
            cur_cash = self.current_asset['cash']
            cur_stock = self.current_asset['stock']
            trading_amount = cur_stock
            
            self.current_asset['cash'] = cur_cash + trading_amount * self.current_trading_price*(1-self.trading_feeRate)
            self.current_asset['stock'] = 0

            self.reward = (- price_change-score_change) *trading_amount

        
        elif self.actions[action] == "Hold":
            self.reward = -1

        else:
            pass

        self.current_asset_value = self.current_asset['stock']*self.current_trading_price + self.current_asset['cash']

        self.defineState()

        self.currentIndex += 1

        if self.currentIndex >= self.finalIndex\
                or self.current_asset_value < self.sudden_death_rate * self.startAssetValue:

            self.done = True

        return self.state, self.reward, self.done, \
               {"trading_feeRate": self.trading_feeRate,\
                "current_trading_price": self.current_trading_price, \
                "current_asset":self.current_asset, "cur_reward": self.reward,
                "act":self.actions[action], "last_asset_value":self.last_asset_value, "current_asset_value":self.current_asset_value}


    def _reset(self,code_stocks):

        self.current_asset = {'cash':self.startAssetValue, 'stock':0}

        self.targetCode= code_stocks

        self.stock_data = self.dataCollect[self.targetCode]

        if self.test == True:
            self.currentIndex = 380
        self.currentIndex = 10

        self.done = False
        self.reward = 0

        self.current_trading_price = self.stock_data[self.currentIndex]

        self.current_asset_value = self.startAssetValue

        self.defineState()


    def _render(self):
        return self.state


    def defineState(self):
        tmpState = []

        self.current_trading_price = self.stock_data[self.currentIndex][5]

        cashFeature = self.current_asset['cash']/self.startAssetValue
        stockFeature = self.current_asset['stock'] * self.current_trading_price/self.startAssetValue
        
        tmpState.append([[cashFeature, stockFeature]])
        
        openpart = []
        highpart = []
        lowpart = []
        closepart = []
        volumnpart = []
        vixpart = []


        for i in range(self.range):
            openpart.append([self.stock_data[self.currentIndex-1-i][0]])
            highpart.append([self.stock_data[self.currentIndex-1-i][1]])
            lowpart.append([self.stock_data[self.currentIndex-1-i][2]])
            closepart.append([self.stock_data[self.currentIndex-1-i][3]])
            volumnpart.append([self.stock_data[self.currentIndex-1-i][4]])
            vixpart.append([self.stock_data[self.currentIndex-1-i][6]])

        tmpState.append([[openpart, highpart, lowpart,closepart, volumnpart, vixpart]])

        tmpState = [np.array(i) for i in tmpState]

        self.state = tmpState


