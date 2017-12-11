'''
Created on Nov 7, 2017

@author: payson
'''
import numpy as np
import os
import pandas as pd

currency_dict={
    "eur":" European Currency",
    "jpy":"Japanese yen",
    "gbp":"Greate Britain Pound",
    "chf":"Swiss Franc",
    "hkd":"Hong Kong Dollar",
    "aud":"Australia Dollar",
    "cad":"Canadian Dollar",
    "nzd":"New Zealand Dollar",
    "rub":"Russia Ruble",
    "krw":"Korean Won",
    "thb":"Thiland Baht",
    "sgd":"Singapore Dollar"
    }
def getCurrencyPairs(mainCurrency):
    pairs=[]      
    for key in currency_dict.keys():
        if key != mainCurrency:
            newPair='fx_s'+mainCurrency+key
            pairs.append(newPair)
    return pairs

def getCurrencyPair(currency,baseCurrency="usd"):
    pair='fx_s'+baseCurrency+currency
    return pair
init_fund_id="usd"
states_dict={
    0 :"usd",
    1 :"eur",
    2 :"jpy",
    3 :"gbp",
    4 :"chf",
    5 :"hkd",
    6 :"aud",
    7 :"cad",
    8 :"nzd",
    9 :"rub",
    10:"krw",
    11:"thb",
    12:"sgd"
    }


inv_states_dict=dict(zip(states_dict.values(),states_dict.keys()))
state_dict_len=len(states_dict)

init_states=np.array([0]*state_dict_len,np.float64).reshape((1,state_dict_len))
init_states[0,inv_states_dict[init_fund_id]]=1

init_reward=np.array([0.002]*state_dict_len,np.float32)
#print(init_state)
init_states_states=np.array([[1/len(states_dict)]*state_dict_len]*state_dict_len,np.float32).T


init_fund_amount=np.array([0]*state_dict_len)
init_fund_amount[inv_states_dict[init_fund_id]]=1000000
#init_fund_amount=np.concatenate((np.array([1000000],np.float64),np.array([0]*12,np.float64)))

forex_fee=0.0025

class Daykline():
    def __init__(self):
        self.data_path="../data"
        self.daykline_data=None
        self.pairs=getCurrencyPairs('usd')
    def load_data(self):
        cvs_path=os.path.join(self.data_path,"daykline.csv")
        self.daykline_data= pd.read_csv(cvs_path)
        return self.daykline_data
    def get_pair_currencies_diff(self,diff=1):
        pair_currencies=[self.daykline_data[self.daykline_data["pair"]==pair] for pair in self.pairs]
        all_pair_currencies_diff1=[]
        for pair_currency in pair_currencies:
            key=pair_currency.iloc[1:,:2]
            content=pair_currency.iloc[:,2:]
            content_diff1=content.diff(diff,axis=0)[1:]
            frames=[key,content_diff1]
            pair_currencies_diff1=pd.concat(frames,axis=1)
            all_pair_currencies_diff1.append(pair_currencies_diff1)
            #print(pair_currencies_diff1)
        return all_pair_currencies_diff1
    def get_pair_currencies(self,pairs=None):
        if pairs==None:
            self.currency_pair=[self.daykline_data[self.daykline_data["pair"]==pair] for pair in self.pairs]
        else:
            self.currency_pair=[self.daykline_data[self.daykline_data["pair"]==pair] for pair in pairs] 


#daykline_data=Daykline()       
#daykline_data.load_data()

#a=[getCurrencyPair(states_dict[i])+"_type" for i in range(len(states_dict))]












