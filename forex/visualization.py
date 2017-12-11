'''
Created on Dec 1, 2017

@author: payson
'''

import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../util")
import util
#rootPath="/media/payson/hdd1/tensorflow/epochs-5/"
rootPath="/home/payson/Documents/eclipse_workspace/python3/ReinforcementInvesting/forex/predictions/"
for i in range(util.state_dict_len-1):
    curr_id=util.states_dict[i+1]
    predictions=pd.read_csv(rootPath+"fx_susd"+curr_id+"predictions.csv",header=None)
    rewards=pd.read_csv(rootPath+"fx_susd"+curr_id+"rewards.csv",header=None)
    begin=100
    end=-1
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(predictions[begin:end].as_matrix(),'k',label='predictions')
    ax.plot(rewards[begin:end].as_matrix(),'k--',label='rewards')
    print(predictions[begin:end].as_matrix())
    print(rewards[begin:end].as_matrix())
    plt.show()
    break
'''
predictions=pd.read_csv("/home/payson/Documents/eclipse_workspace/python3/ReinforcementInvesting/forex/fx_susdjpypredictions.csv",header=None)
rewards=pd.read_csv("/home/payson/Documents/eclipse_workspace/python3/ReinforcementInvesting/forex/fx_susdjpyrewards.csv",header=None)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(predictions[2100:2700].as_matrix()/2,'k',label='predictions')
ax.plot(rewards[2100:2700].as_matrix(),'k--',label='rewards')
plt.show()
'''