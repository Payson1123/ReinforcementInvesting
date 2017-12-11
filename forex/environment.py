'''
Created on Nov 7, 2017

@author: payson
'''
import numpy as np
import numpy.random as rnd
import pandas as pd
import sys
import os
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from collections import deque


sys.path.append("../util")
import util
import data

rnd.seed(42)

class environment():
    def __init__(self,config,time_begin,time_end=None,cross_price=False,is_test=False,state=None,amount=None):

        #self.curr_state=pd.Series(rnd.randn(util.state_dict_len))
        #self.next_state=pd.Series(rnd.randn(util.state_dict_len))
        if amount==None:
            self.curr_amount=util.init_fund_amount
        self.done=False
        self.is_test=is_test
        self.eps_min=0.05
        self.eps_max=1.0
        self.eps_decay_steps=50000 
        self.step_num=0
        self.num_steps=0
        self.preactions=deque([],maxlen=self.num_steps)
        self.reward=0
        self.checkpoint_path="./param/env.ckpt"
        self.state_state_probability=util.init_states_states
        self.config=config          
        self.daykline=data.Daykline()
        self.daykline.load_data()
        #self.price_pct_change=self.daykline.get_all_price_pct_change()["close"][time_begin:].fillna(0).as_matrix()
        if not time_end:
            self.price_pct_change=self.daykline.get_all_cross_prices_pct_diff()["close_close_pct"][time_begin:].fillna(0).as_matrix()
        else:
            self.price_pct_change=self.daykline.get_all_cross_prices_pct_diff()["close_close_pct"][time_begin:time_end].fillna(0).as_matrix()
        #self.all_cross_prices_pct_diff=self.daykline.get_all_cross_prices_pct_diff()["close"][time_begin:].fillna(0).as_matrix()
        self.max_step_num=len(self.price_pct_change)
        self.split_train_test_set()
        #self.all_price_pct_change_in_khb=self.daykline.get_all_price_pct_change_in_khb()[time_begin:]
        #self.all_price_pct_change_in_khb=self.daykline.get_all_cross_price_pct_diff_in_khb()[time_begin:]
        #print(self.all_price_pct_change_in_khb.shape)
        #self.all_price_pct_change_in_khb_array=self.all_price_pct_change_in_khb.as_matrix()
        '''
        if cross_price:
            self.all_price_pct_change_in_khb=self.daykline.get_all_cross_price_pct_diff_in_khb()[time_begin:]         
            self.all_price_pct_change_in_khb_array=self.compress(self.all_price_pct_change_in_khb.as_matrix())
        else:
            self.all_price_pct_change_in_khb=self.daykline.get_all_price_pct_change_in_khb()[time_begin:]
            self.all_price_pct_change_in_khb_array=self.all_price_pct_change_in_khb.as_matrix()
        '''
        
        self.all_price_pct_change_in_khb=self.daykline.get_all_currency_pair_ave_pct_diff_pivot()[time_begin:]
        self.all_price_pct_change_in_khb_array=self.all_price_pct_change_in_khb.as_matrix()
        
        
        (self.r,self.c)=self.all_price_pct_change_in_khb.shape
        
        '''
        max_input_num=self.r-self.config.num_steps+1       
        self.inputs=np.array([self.all_price_pct_change_in_khb_array[i:i+self.config.num_steps] for i in range(max_input_num)])
        for i in range(max_input_num):
            self.inputs.append(self.all_price_pct_change_in_khb[i:i+self.config.num_steps])
        print(len(self.inputs))
        print(self.inputs)
        self.inputs=np.array(self.inputs)
        '''
            
        self.reward=1
        self.curr_action=0
        self.pre_action=None  

        self.sample_curr_action=0
        self.sample_pre_action=None     
    def get_input(self,step):
        return self.all_price_pct_change_in_khb_array[step:step+self.config.num_steps]
    def compress(self,all_price_pct_change_in_khb):   
        print("compressing")
        r,c=all_price_pct_change_in_khb.shape
        print(r,c)
        inputs=tf.placeholder(tf.float64,shape=[None,c])
        l2_reg=0.002
        learning_rate=0.002
        n_hidden1=c//2
        n_hidden2=c//3
        n_hidden3=c//4
        n_hidden4=n_hidden2
        n_hidden5=n_hidden1
        n_outputs=c
        n_epoch=500
        batch_size=50
        print(self.checkpoint_path)
        with tf.contrib.framework.arg_scope([fully_connected],
                                            activation_fn=tf.nn.elu,
                                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)):
            
            hidden1=fully_connected(inputs,n_hidden1)
            hidden2=fully_connected(hidden1,n_hidden2)
            hidden3=fully_connected(hidden2,n_hidden3)
            hidden4=fully_connected(hidden3,n_hidden4)
            hidden5=fully_connected(hidden4,n_hidden5)           
            outputs=fully_connected(hidden5,n_outputs,activation_fn=None)        
                
        rebuild_loss=tf.reduce_mean(tf.square(outputs-inputs))
        regression_loss=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss=tf.add_n([rebuild_loss]+regression_loss)
        optimizer=tf.train.AdamOptimizer(learning_rate)
        train_op=optimizer.minimize(loss)
        
        init=tf.global_variables_initializer()
        saver=tf.train.Saver()
        pre_loss=0
        loss_val=0
        with tf.Session() as sess:
            if os.path.isfile(self.checkpoint_path):
                self.saver.restore(self.checkpoint_path)           
                loss_val=rebuild_loss.eval(feed_dict={inputs:all_price_pct_change_in_khb})
                print("loss:",loss_val) 
                feature_inouts=hidden3.eval(feed_dict={inputs:all_price_pct_change_in_khb})
                outputs_val=outputs.eval(feed_dict={inputs:all_price_pct_change_in_khb})
                print("inouts:",all_price_pct_change_in_khb[0:10])
                print("feature_inouts:",feature_inouts[0:10])
                print("outputs_val:",outputs_val[0:10])
                return feature_inouts
            else:
                init.run()         
                for epoch in range(n_epoch):
                    indices=rnd.permutation(r)
                    for batch in range(r//batch_size):
                        #batch_indices=np.random.randint(r,batch_size)
                        batch_indices=indices[batch*batch_size:(batch+1)*batch_size]                   
                        batch_inputs=all_price_pct_change_in_khb[batch_indices]
                        sess.run(train_op,feed_dict={inputs:batch_inputs})
                        #print(feature_inouts[0])
                    pre_loss= loss_val 
                    loss_val=rebuild_loss.eval(feed_dict={inputs:all_price_pct_change_in_khb})
                    
                    print("epoch:",epoch,"loss:",loss_val)
                    '''
                    if loss_val<10 or (loss_val-pre_loss)<10:
                        break
                    '''
                    saver.save(sess,self.checkpoint_path)
                
                feature_inouts=hidden3.eval(feed_dict={inputs:all_price_pct_change_in_khb})
                outputs_val=outputs.eval(feed_dict={inputs:all_price_pct_change_in_khb})
                print("inouts:",all_price_pct_change_in_khb[0:10])
                print("feature_inouts:",feature_inouts[0:10])
                print("outputs_val:",outputs_val[0:10])
                return feature_inouts
           
    def reset(self,num_steps=0,is_test=False):
        self.num_steps=num_steps
        self.preactions=deque([],maxlen=self.num_steps)
        if is_test:
            self.step_num=self.max_step_num
        else:
            self.step_num=num_steps
        self.done=False
        if num_steps==0:
            self.curr_state=self.all_price_pct_change_in_khb_array[self.step_num]
        else:
            self.curr_state=self.all_price_pct_change_in_khb_array[self.step_num-self.num_steps:self.step_num] 
        #self.next_state=self.all_price_pct_change_in_khb_array[self.step_num+1]        
    def get_max_step_num(self):
        return self.max_step_num

    def split_train_test_set(self,ratio=0.01):
        self.max_step_num=self.max_step_num*(1-ratio)
        
    def sample_action(self):          
        sample_value=np.random.random_sample()
        accumulated_value=0
        sample_index=0       
        for item in self.state_state_probability[0]:
            accumulated_value+=item
            if accumulated_value>sample_value:
                break
            sample_index+=1
        
        self.sample_pre_action=self.sample_curr_action
        self.sample_curr_action=sample_index
        return self.sample_pre_action,self.sample_curr_action
    def onehot_array(self,idx,n):
        temp=np.zeros(n)
        temp[idx]=1
        return temp
    def onehot_matrix(self,idx,n):
        size=len(idx)
        temp=[self.onehot_array(idx[i],n) for i in range(size)]
        temp=np.asarray(temp)
        return temp
    def leap_forward(self,first_action,leap=3,discount_rate=0.9999,isTest=False,transaction_fee=None,pre_action=None): 
            
        step_num=self.step_num 
        preactions=self.preactions
        curr_state=self.curr_state
        action=first_action
        reward=1
        for step in step_num(leap):
            step_num+1
            preactions.append(action)
            if step_num<self.max_step_num:
                if self.num_steps==0:
                    curr_state=self.all_price_pct_change_in_khb_array[step_num-1:step_num]
                    onehotmat=self.onehot_matrix(tuple(action),util.state_dict_len)
                    curr_state=np.concatenate((curr_state,onehotmat),axis=1)
                    #self.real_reward=np.max(self.price_pct_change[self.step_num],axis=1)
                else:
                    curr_state=self.all_price_pct_change_in_khb_array[step_num-self.num_steps:step_num]
                    random_size=self.num_steps-len(preactions)
                    random_actions=np.array([[]])
                    if random_size>0:
                        radom_idx=np.random.randint(util.state_dict_len,size=random_size)
                        random_actions=self.onehot_matrix(tuple(radom_idx),util.state_dict_len)
                    #print(self.preactions)
                    onehotmat=self.onehot_matrix(tuple(preactions),util.state_dict_len)
                    if len(random_actions[0])>0:
                        if len(preactions)>0:
                            onehotmat=np.concatenate((random_actions,onehotmat),axis=0)
                        else:
                            onehotmat=random_actions
                    curr_state=np.concatenate((curr_state,onehotmat),axis=1)
                    
                max_reward=np.max(self.price_pct_change[step_num],axis=0)
                action=np.argmax(self.price_pct_change[step_num],axis=0)
                reward=reward*discount_rate*(1+max_reward)

        return curr_state,reward  
    
    def step_forward(self,action,isTest=False,transaction_fee=None,pre_action=None): 
        
        self.step_num+=1 
        self.preactions.append(action) 
        if self.step_num<self.max_step_num:
            if self.num_steps==0:
                self.curr_state=self.all_price_pct_change_in_khb_array[self.step_num-1:self.step_num]
                onehotmat=self.onehot_matrix(tuple(action),util.state_dict_len)
                self.curr_state=np.concatenate((self.curr_state,onehotmat),axis=1)
                #self.real_reward=np.max(self.price_pct_change[self.step_num],axis=1)
            else:
                self.curr_state=self.all_price_pct_change_in_khb_array[self.step_num-self.num_steps:self.step_num]
                random_size=self.num_steps-len(self.preactions)
                random_actions=np.array([[]])
                if random_size>0:
                    radom_idx=np.random.randint(util.state_dict_len,size=random_size)
                    random_actions=self.onehot_matrix(tuple(radom_idx),util.state_dict_len)
                #print(self.preactions)
                onehotmat=self.onehot_matrix(tuple(self.preactions),util.state_dict_len)
                if len(random_actions[0])>0:
                    if len(self.preactions)>0:
                        onehotmat=np.concatenate((random_actions,onehotmat),axis=0)
                    else:
                        onehotmat=random_actions
                self.curr_state=np.concatenate((self.curr_state,onehotmat),axis=1)
                #self.real_reward=np.max(self.price_pct_change[self.step_num],axis=1)
            #self.next_state=self.all_price_pct_change_in_khb_array[self.step_num+1]
            next_currency=util.states_dict[action]   
            currency_pair=util.getCurrencyPair(next_currency)
                
            #self.reward=self.price_pct_change[self.step_num][action]-self.price_pct_change[self.step_num-1][action]
            if pre_action:
                #self.reward=self.price_pct_change[self.step_num][action]-self.price_pct_change[self.step_num][pre_action]
                if not transaction_fee:
                        self.reward=self.price_pct_change[self.step_num-1][action]
                        self.max_reward=np.max(self.price_pct_change[self.step_num-1],axis=0)
                else:
                    if pre_action==action:
                        self.reward=self.price_pct_change[self.step_num-1][action]
                        self.max_reward=np.max(self.price_pct_change[self.step_num],axis=0)
                    else:
                        self.reward=self.price_pct_change[self.step_num-1][action]-transaction_fee
                        self.max_reward=np.max(self.price_pct_change[self.step_num-1],axis=0)-transaction_fee
                #print("reward:",self.reward)
            else:
                self.reward=self.price_pct_change[self.step_num-1][action]    
                self.max_reward=np.max(self.price_pct_change[self.step_num-1],axis=0)      
            return self.reward,self.max_reward,self.curr_state,currency_pair,self.done
        else:
            if self.is_test:
                return
            else:
                self.done=True
                return None,None,None,None,self.done
      
    
    def get_price_pct_change(self,curr_index):
        if self.num_steps+curr_index>self.max_step_num-1:
            return np.array([])
        else:
            return np.array([self.price_pct_change[self.num_steps+curr_index]])
    def get_max_price_pct_change(self,curr_index):
        
        return np.array([np.max(self.price_pct_change[curr_index])])   
    def __str__(self):
        rlt=''
        for amt in self.curr_amount:
            rlt+=str(amt)+','
        return rlt[:-1]
    
if __name__ == '__main__':
    env=environment(time_begin="2000-10",config=None)
    