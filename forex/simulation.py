'''

This class is used to simulate the process of buying or selling foreign currency with specified action probability

Created on Nov 7, 2017

@author: payson
'''
import numpy as np
import pandas as pd
import numpy.random as rnd
import tensorflow as tf
import tensorflow.contrib.factorization.python.ops.gmm
import sys
import os


sys.path.append("../util")
import util
import data
import environment 

class Config(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 25
  #hidden_size=675
  #hidden_size = 1225
  hidden_size=624
  hidden_size1 = 1238
  #hidden_size = 2701
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 1
  pairs_num=13
  
  
def data_type():
  return  tf.float32  
class montecarlo():
    def __init__(self):
        self.states=util.init_states
        self.states_states=util.init_states_states
        self.reward=[]   
        self.config=Config()
        self.step=0
        self.eps_min=0.05
        self.eps_max=1.0        
        self.eps_decay_steps=50000 
        self.checkpoint_path="./mydqn2.ckpt"
        
    def rnn_network(self,batch_input,num_steps=None,scope="actor",is_training=True):  
        #print(batch_input)
        #cell=tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=0.0, state_is_tuple=True,reuse=False)
 
        if "actor" in scope:
            batch_size=1
        else:
            batch_size=self.config.batch_size
        '''
        if is_training and self.config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=self.config.keep_prob)
        '''

        outputs=[]   
        if not num_steps:
            num_steps=self.config.num_steps
        with tf.variable_scope(scope) as scope:
            cell=tf.contrib.rnn.LSTMBlockCell(self.config.hidden_size, forget_bias=0.0)
            cells=[cell for _ in range(self.config.num_layers)]
            multiCell=tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
            self._initial_state = multiCell.zero_state(batch_size,data_type())  
            state = self._initial_state           
            for time_step in range(num_steps):         
                if is_training and time_step > 0: tf.get_variable_scope().reuse_variables()    
                if not is_training:  tf.get_variable_scope().reuse_variables()                    
                (cell_output, state) = multiCell(batch_input[:, time_step, :], state)
                
            #only the last step is valuable
            outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.config.hidden_size])
        return output, state   
 
         
    def multi_rnn_network(self,batch_input,num_steps,scope="actor",is_training=True):  
        #print(batch_input)
        #cell=tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=0.0, state_is_tuple=True,reuse=False)
 
        if "actor" in scope:
            batch_size=1
        else:
            batch_size=self.config.batch_size
        cell=tf.contrib.rnn.LSTMBlockCell(self.config.hidden_size, forget_bias=0.0)
        '''
        if is_training and self.config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=self.config.keep_prob)
        '''
        cells=[cell for _ in range(self.config.num_layers)]
        multiCell=tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
        self._initial_state = multiCell.zero_state(batch_size,data_type())
        state = self._initial_state
        outputs=[]   
        with tf.variable_scope(scope) as scope:
            for time_step in range(num_steps):         
                if is_training and time_step > 0: tf.get_variable_scope().reuse_variables()    
                if not is_training:  tf.get_variable_scope().reuse_variables()                    
                (cell_output, state) = multiCell(batch_input[:, time_step, :], state)
            #only the last step is valuable
            outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.config.hidden_size])
        
        
        
        return output, state    
    
    def process_old(self,epochs=1,steps=1,time_begin="2010-10"):
        '''
        self.daykline.load_data()
        all_price_pct_change_in_khb=self.daykline.get_all_price_pct_change_in_khb()[time_begin:]
        
        curr_state=np.argmax(self.states)
        print(curr_state)
        print(all_price_pct_change_in_khb.shape)
        all_price_pct_change_in_khb_array=all_price_pct_change_in_khb.as_matrix()
        '''
        temp=[1]*(703*1225)
        temp=np.array(temp).reshape(703,1225)
        all_price_pct_change_in_khb=temp
        (r,c)=all_price_pct_change_in_khb.shape
        inputs=[]
        for i in range(r-self.config.num_steps+1):
            inputs.append(all_price_pct_change_in_khb[i:i+self.config.num_steps])
        inputs=np.array(inputs)
        
     
        batch_index=np.random.randint(r-self.config.num_steps+1,size=self.config.batch_size)       
        batch_input=inputs[batch_index]
        batch_y=inputs[batch_index+1]
        batch_input=batch_input.reshape((self.config.batch_size,self.config.num_steps,self.config.hidden_size))
        print("batch_input.shape")
        print(batch_input.shape)       
        
        (output, state)=self.q_network(batch_input)
        
        softmax_w = tf.get_variable("softmax_w",[self.config.hidden_size,self.config.pairs_num],dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.config.pairs_num], dtype=data_type())
        
        logits=tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        logits = tf.reshape(logits, [self.config.batch_size, 1, self.config.pairs_num])
        outputs=tf.sigmoid(logits)
        
        '''
        all_price_pct_change=self.daykline.get_all_price_pct_change()[type][time_begin:]
        print(all_price_pct_change)
        
        for epoch in range(epochs):
            for step in range(steps):
                next_state=self.get_next_state()
                column_name=util.states_dict[next_state]
    
        '''    
    def epsilon_greedy(self,q_values,step):
        epsilon=max(self.eps_min,self.eps_max-(self.eps_max-self.eps_min)*step/self.eps_decay_steps)
        if rnd.rand()<epsilon:
            curr_action=rnd.randint(self.config.pairs_num)
        else:
            curr_action=np.argmax(q_values)   
        return curr_action  
    def action_sampling(self,prob):
        s = np.random.uniform(0,1,1)
        sum=0
        for (idx,p) in enumerate(prob):
            sum+=p
            if sum>=s:
                return idx
        
    def epsilon_softmax(self,q_values,step):
        epsilon=max(self.eps_min,self.eps_max-(self.eps_max-self.eps_min)*step/self.eps_decay_steps)

        if rnd.rand()<epsilon:
            curr_action=rnd.randint(self.config.pairs_num)
        else:        
            prob=np.exp(q_values)/np.sum(np.exp(q_values))
            indice=np.argsort(prob)
            sort_prob=prob[-1,np.argsort(prob)]
            idx=self.action_sampling(sort_prob[0])
            
            curr_action=indice[0][idx]
            #curr_action=np.argmax(q_values)   
        return curr_action           
    
    def get_sample_rewards(self,env,num=5000):
        print(util.state_dict_len,env.get_max_step_num(),util.state_dict_len)
        rewards=np.zeros((env.get_max_step_num()-1,util.state_dict_len))     
        for i in range(num):           
            for first_action in range(util.state_dict_len):
                env.reset()
                pre_action,action=env.sample_action()
                step=0
                reward,max_reward,curr_state,currency_pair,done=env.step_forward(action)
                rewards[step][action]=reward
                while True:     
                    step+=1               
                    pre_action,action=env.sample_action()
                    reward,max_reward,curr_state,currency_pair,done=env.step_forward(action,pre_action)  
                    #print(action)           
                    if done:
                        break
                    rewards[step][action]=reward    
        return rewards*util.state_dict_len/num
    
    def rewards_accumulate(self,rewards,step_num=None): 
        #if step is not None, reward will accumulate from step_num forward step
        if not step_num:
            reward_idx,action_idx=rewards.shape
            print(reward_idx,action_idx)
            best_actions=np.zeros(reward_idx)
            max_rewards=np.ones((reward_idx,action_idx))
            
            for i in range(reward_idx):
                for j in range(action_idx):
                    action_reward=[(max_rewards[-i][k])*(1+rewards[-(i+1)][j]) for k in range(action_idx)]
                    max_rewards[-(i+1)][j]=np.max(action_reward)                  
                    best_actions[-(i+1)]=np.argmax(max_rewards[-(i+1)])
            return  max_rewards,best_actions     
        else:
            reward_idx,action_idx=rewards.shape
            print(reward_idx,action_idx)
            if step_num>reward_idx:
                step_num=reward_idx
            best_actions=np.zeros(step_num)
            max_rewards=np.ones((step_num,action_idx))
            for i in range(step_num):
                for j in range(action_idx):
                    action_reward=[(max_rewards[-i][k])*(1+rewards[-(i+1)][j]) for k in range(action_idx)]
                    max_rewards[-(i+1)][j]=np.max(action_reward)                  
                    best_actions[-(i+1)]=np.argmax(max_rewards[-(i+1)])
            return  max_rewards,best_actions                      
            
    def pgm_process(self,epochs=1,steps=1,time_begin="2010-10"):
        #Using probabilitic graphical model to process
        self.env=environment.environment(self.config,time_begin)
        sample_rewards=self.get_sample_rewards(self.env)
        print(sample_rewards)
        max_rewards,best_actions=self.rewards_accumulate(sample_rewards)
        print(max_rewards[0])
        print(best_actions)
    def test_rl_process(self,time_begin="2010-10"):
        env=environment.environment(self.config,time_begin)
        with tf.Session() as sess:
            if os.path.isfile(self.checkpoint_path):
                self.saver.restore(self.checkpoint_path)
                env.reset(num_steps=self.config.num_steps,is_test=True)
                
            else:
                print("no trained parameters")
            
    def rl_process(self,lookforward_steps=1,epochs=1,steps=1,time_begin="2010-10"):
        #Using Reinforcement Learning to process
        env=environment.environment(self.config,time_begin)
        learning_rate=0.001
        discount_rate=0.9999
        iteration=2000
        memory_reset_steps=20
        leap=3
        print(util.state_dict_len,env.get_max_step_num(),util.state_dict_len)
        #rewards=np.zeros((env.get_max_step_num()-1,util.state_dict_len))   
        X=tf.placeholder(tf.float32,shape=(None,self.config.num_steps,self.config.hidden_size1),name="batch_input")                                        
        #y=tf.placeholder(tf.float32,shape=(None,self.config.pairs_num))      
            
        
        (critic_output, critic_state)=self.rnn_network(X,"q_networks",True)        
        softmax_w = tf.get_variable("softmax_w",[self.config.hidden_size1,self.config.pairs_num],dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.config.pairs_num], dtype=data_type())       
        critic_logits=tf.nn.xw_plus_b(critic_output, softmax_w, softmax_b)              
        critic_q_value=tf.reshape(critic_logits, [self.config.batch_size, self.config.pairs_num])
        
        (actor_output, actor_state)=self.rnn_network(X,"q_networks",False)
        actor_logits=tf.nn.xw_plus_b(actor_output, softmax_w, softmax_b)      
        actor_q_value = tf.reshape(actor_logits, [1, self.config.pairs_num])
        
        with tf.variable_scope("train") as train_scope:
            X_action=tf.placeholder(tf.int32,shape=[None])
            y=tf.placeholder(tf.float32,shape=[None,1])
            #batch_q_value=np.max(critic_q_value,axis=1)
            batch_q_value=tf.reduce_sum(critic_q_value*tf.one_hot(X_action,self.config.pairs_num),axis=1,keep_dims=True)
            cost=tf.reduce_mean(tf.square(y-batch_q_value))
            global_step=tf.Variable(0,trainable=False,name="global_step")
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op=optimizer.minimize(cost,global_step=global_step)            
        init=tf.global_variables_initializer()
        self.saver=tf.train.Saver()
        np.random.seed(100)
        with tf.Session() as sess:
            if os.path.isfile(self.checkpoint_path):
                self.saver.restore(self.checkpoint_path)
            else:
                init.run()       
            #generate batch data set  
            critic_step=0  
            #actor_step=0
            for i in range(iteration):    
                states=[]  
                q_values=[] 
                actions=[]     
                rewards=[]
                max_rewards=[]
                env.reset(num_steps=self.config.num_steps)
                pre_action,action=env.sample_action()
                actions.append(action)
                actor_step=global_step.eval()
                reward,max_reward,curr_state,currency_pair,done=env.step_forward(action)
                states.append(curr_state)
                curr_state=curr_state.reshape((1,self.config.num_steps,self.config.hidden_size1))    
                internal_state,internal_reward=env.leap_forward(action,leap)
                q_value=actor_q_value.eval(feed_dict={X:internal_state})
                q_value=reward*np.power(discount_rate,leap+1)*q_value
                q_values.append(q_value)
                
                next_action=self.epsilon_greedy(q_value, actor_step)

                rewards.append(reward)
                max_rewards.append(max_reward)
                while True:     
                    pre_action=action
                    action= next_action 
                    actions.append(action) 
                    #print(action)                      
                    reward,max_reward,curr_state,currency_pair,done=env.step_forward(action,pre_action)
                    #print("reward:",reward)
                    if done:
                        break  
                    rewards.append(reward)
                    max_rewards.append(max_reward)
                    states.append(curr_state)
                    curr_state=curr_state.reshape((1,self.config.num_steps,self.config.hidden_size1))    
                    
                    q_value=actor_q_value.eval(feed_dict={X:curr_state})
                    q_values.append(q_value)
                    next_action=self.epsilon_greedy(q_value, actor_step)
                    #print("action:",next_action)
                    #print(next_action,step,done)         
                next_states=np.asarray(states[1:])  
                states=np.asarray(states)  
                q_values=np.asarray(q_values)
                actions=np.asarray(actions)    
                rewards=np.asarray(rewards)
                max_rewards=np.asarray(max_rewards)
                #print("states.shape",states.shape)
                print("rewards:",np.prod(rewards+1))
                
                #print("max_rewards:",max_rewards+1)
                #print("max_total_rewards:",np.prod(max_rewards+1))
                
                #rewards=rewards*10000   
                #training   
                if critic_step+memory_reset_steps>len(states)-1:
                    critic_begin_step=critic_step
                    critic_end_step=len(states)-1
                    critic_step=0
                else:
                    critic_begin_step=critic_step
                    critic_end_step=critic_begin_step+memory_reset_steps
                    critic_step=critic_end_step
                for j in range(critic_begin_step,critic_end_step):                                   
                    #indices=rnd.permutation(len(states))[:self.config.batch_size]
                    #batch_indices=np.random.randint(len(states)-1,size=self.config.batch_size)
                    #using TD(step) algorithm
                    y_batch=1
                    batch_actions=0
                    batch_states=np.asarray(states[0])  
                    for step in range(lookforward_steps):
                        #To ensure next_states[batch_indices] not out of bound,   j+step should not larger than len(states)-2
                        if j+step>len(states)-2:
                            break
                        batch_indices=np.array([j+step])                                       
                        #print("batch_next_states.shape:",batch_next_states.shape)
                        if step==0:
                            batch_actions=np.asarray(actions[batch_indices])
                            batch_states=np.asarray(states[batch_indices])      
                        #batch_q_values=np.asarray(q_values[batch_indices])
                        batch_rewards=np.asarray(rewards[batch_indices])
                        y_batch=y_batch*discount_rate*(1+batch_rewards)
                        if step==lookforward_steps-1:
                            #batch_next_states=np.asarray(next_states[batch_indices]).reshape(self.config.batch_size,self.config.num_steps,self.config.hidden_size)
                            batch_states=np.asarray(states[batch_indices])  
                            batch_next_q_value=critic_q_value.eval(feed_dict={X:batch_states})
                            batch_max_next_q_value=np.max(batch_next_q_value,axis=1)
                            #the reward is a ratio, so using the following formula as reward accumulating reward
                            y_batch=y_batch*discount_rate*(1+batch_max_next_q_value)

                        
                    y_batch=y_batch.reshape(-1,1)
                    train_op.run(feed_dict={y:y_batch,X:batch_states,X_action:batch_actions})
                    '''
                    if j== memory_reset_steps-1:
                        cost_val=cost.eval(feed_dict={y:y_batch,X:batch_states,X_action:batch_actions}) 
                        print("index:",j)
                        print("y:",y_batch)
                        print("q_value:",batch_q_value.eval(feed_dict={X:batch_states,X_action:batch_actions}) )
                        print("cost_val:",cost_val)
                    '''    
                self.saver.save(sess,self.checkpoint_path)
                print("epoch:",i) 
        

    def lstm_process(self,epochs=1,steps=1,time_begin="2000-10"):
        #Using LSTM to predict value
        self.env=environment.environment(self.config,time_begin)
        X=tf.placeholder(tf.float32,shape=(None,self.config.num_steps,self.config.hidden_size),name="batch_input")                                        
        y=tf.placeholder(tf.float32,shape=(None,self.config.pairs_num))         
        (output, state)=self.rnn_network(X)        
        softmax_w = tf.get_variable("softmax_w",[self.config.hidden_size,self.config.pairs_num],dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.config.pairs_num], dtype=data_type())       
        logits=tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        q_values = tf.reshape(logits, [self.config.batch_size, self.config.pairs_num])
        
        #xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=q_values)
        loss=tf.reduce_mean(tf.square(y-q_values))
        
        learning_rate=0.005
        with tf.name_scope("train") as scope:
            optimizer=tf.train.GradientDescentOptimizer(learning_rate)
            train_op=optimizer.minimize(loss)       
        init=tf.global_variables_initializer()  
         
        with tf.Session() as sess:
            init.run()           
            while True:
                if self.step>0:
                    tf.get_variable_scope().reuse_variables()
                
                input_batch=self.env.get_input(self.step)

                y_batch=self.env.get_price_pct_change(self.step)        
                input_batch=input_batch.reshape((self.config.batch_size,self.config.num_steps,self.config.hidden_size))                     
                sess.run(train_op,feed_dict={X:input_batch,y:y_batch})
                loss_train=loss.eval(feed_dict={X:input_batch,y:y_batch})
                if(self.step%100==0):
                    print(" train loss:",loss_train) 
                    
                '''          
                action=self.env.epsilon_greedy(q_values,curr_index)
                reward=self.env.step_forward(action,curr_index)
                '''
                
                            
                '''
                all_price_pct_change=self.daykline.get_all_price_pct_change()[type][time_begin:]
                print(all_price_pct_change)
                
                for epoch in range(epochs):
                    for step in range(steps):
                        next_state=self.get_next_state()
                        column_name=util.states_dict[next_state]
            
                '''
                self.step+=1    
    def lstm_process_all_currency(self,epochs=20,steps=1,time_begin="2006-10",time_end="2016-11"):
        self.env=environment.environment(self.config,time_begin,time_end)
        for i in range(len(util.states_dict)-1):           
            self.lstm_process_one_currency(self.env,i+1,epochs=epochs)
            break
    def lstm_process_one_currency(self,env,curr_idx,multi_lstm=1,epochs=5):
        #Using LSTM to predict value
        currency_type=util.states_dict[curr_idx]
        scope="fx_susd"+currency_type
        checkpoint_path="./"+scope+".ckpt"
        with tf.variable_scope(scope) as scope:
            X=tf.placeholder(tf.float32,shape=(None,self.config.num_steps,self.config.hidden_size),name="batch_input")                                        
            y=tf.placeholder(tf.float32,shape=(None,1))  
            (output, state)=self.rnn_network(X)         
            for i in range(multi_lstm-1):
                network_scope="actor"+str(i+1)
                output=tf.reshape(output,[-1,1,self.config.hidden_size])
                (output, state)=self.rnn_network(output,num_steps=1,scope=network_scope)  
            softmax_w = tf.get_variable("softmax_w",[self.config.hidden_size,1],dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [1], dtype=data_type())       
            logits=tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            prediction = tf.reshape(logits, [self.config.batch_size, 1])
            
            #xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=q_values)
            loss=tf.reduce_mean(tf.square(y-prediction))
            
            learning_rate=0.0001
            with tf.name_scope("train") as scope:
                optimizer=tf.train.AdamOptimizer(learning_rate)
                train_op=optimizer.minimize(loss)       
            init=tf.global_variables_initializer()  
            saver=tf.train.Saver()
            cost_discout=0.999
            with tf.Session() as sess:
                if os.path.isfile(checkpoint_path):
                    saver.restore(sess,checkpoint_path)
                else:
                    init.run()  
                                  
                first_step=True 
                predictions=[]
                rewards=[]
                for epoch in range(epochs):
                    env.reset(num_steps=self.config.num_steps)   
                    self.step=0
                    cost=0.0       
                    while True:
                        if not first_step:
                            tf.get_variable_scope().reuse_variables()
                        else:
                            first_step=False                                      
                        input_batch=env.get_input(self.step)
    
                        y_batch=env.get_price_pct_change(self.step)  
                                         
                        if y_batch.size==0:
                            break   
                                          
                        y_batch=y_batch[:,curr_idx]  
                        rewards.append(y_batch)
                        
                        input_batch=input_batch.reshape((self.config.batch_size,self.config.num_steps,self.config.hidden_size))                     
                        y_batch=y_batch.reshape((self.config.batch_size,1))                        
                        
                        prediction_val=prediction.eval(feed_dict={X:input_batch})
                        if epoch==epochs-1:
                            predictions.append(prediction_val)
                            rewards.append(y_batch) 
                        #To make lstm network adaptable to the new environment, the network will tbe trained every day 
                        sess.run(train_op,feed_dict={X:input_batch,y:y_batch})                       
                        loss_train=loss.eval(feed_dict={X:input_batch,y:y_batch})
                        
                        cost+=loss_train
                        
                        self.step+=1
                    print("epoch :",epoch,"cost:",cost) 
                    saver.save(sess,checkpoint_path) 
                
                if len(rewards)>0:
                    np.savetxt("fx_susd"+currency_type+"predictions.csv",np.asarray(predictions),delimiter=",")
                    np.savetxt("fx_susd"+currency_type+"rewards.csv",np.asarray(rewards),delimiter=",")  
                  
                                  
    def lstm_test_all_currency(self,time_begin="2016-11"):
        self.env=environment.environment(self.config,time_begin)
        for i in range(len(util.states_dict)-1):
            self.lstm_test_one_currency(self.env,i+1)
            break
    def lstm_test_one_currency(self,env,curr_idx):
        #Using LSTM to predict value
        currency_type=currency_type=util.states_dict[curr_idx]
        print("currency_type:"+currency_type)
        scope="fx_susd"+currency_type
        #checkpoint_path="/media/payson/hdd1/tensorflow/epochs-5/"+scope+".ckpt"
        checkpoint_path="/home/payson/Documents/eclipse_workspace/python3/ReinforcementInvesting/forex/"+scope+".ckpt"
        with tf.variable_scope(scope) as scope:
            X=tf.placeholder(tf.float32,shape=(None,self.config.num_steps,self.config.hidden_size),name="batch_input")                                        
            y=tf.placeholder(tf.float32,shape=(None,1))         
            (output, state)=self.rnn_network(X)        
            softmax_w = tf.get_variable("softmax_w",[self.config.hidden_size,1],dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [1], dtype=data_type())       
            logits=tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            prediction = tf.reshape(logits, [self.config.batch_size, 1])            
            #xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=q_values)
            loss=tf.reduce_mean(tf.square(y-prediction))
            
            learning_rate=0.0001
            with tf.name_scope("train") as scope:
                optimizer=tf.train.AdamOptimizer(learning_rate)
                train_op=optimizer.minimize(loss)        
            init=tf.global_variables_initializer()  
            saver=tf.train.Saver()
            cost_discout=0.999
            with tf.Session() as sess:
                '''
                if os.path.isfile(checkpoint_path):
                    saver.restore(sess,checkpoint_path)
                else:
                    init.run()  
                '''
                saver.restore(sess,checkpoint_path)                
                first_step=True 
                predictions=[]
                rewards=[]

                env.reset(num_steps=self.config.num_steps)   
                self.step=0
                cost=0.0       
                while True:
                    if not first_step:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        first_step=False                                      
                    input_batch=env.get_input(self.step)

                    y_batch=env.get_price_pct_change(self.step)                   
                    if y_batch.size==0:
                        break   
                 
                    y_batch=y_batch[:,curr_idx]  
                    input_batch=input_batch.reshape((self.config.batch_size,self.config.num_steps,self.config.hidden_size))                     
                    y_batch=y_batch.reshape((self.config.batch_size,1))                        
                    prediction_val=prediction.eval(feed_dict={X:input_batch})

                    predictions.append(prediction_val)
                    rewards.append(y_batch)                        
                    loss_val=loss.eval(feed_dict={X:input_batch,y:y_batch})
                    self.step+=1
                    cost+=loss_val
                    print("cost:",cost) 
                    #To make lstm network adaptable to the new environment, the network will tbe trained every day  
                    sess.run(train_op,feed_dict={X:input_batch,y:y_batch})   
                    #saver.save(sess,checkpoint_path) 
                
                if len(predictions)>0:
                    np.savetxt("./predictions/fx_susd"+currency_type+"predictions.csv",np.asarray(predictions),delimiter=",")
                    np.savetxt("./predictions/fx_susd"+currency_type+"rewards.csv",np.asarray(rewards),delimiter=",")    
    def epsilon_max_idx(self,curr_id):  
        epsilon=0.05
        if rnd.rand()<epsilon:
            curr_action=rnd.randint(12)+1   
            return util.states_dict[curr_action]
        else:
            return curr_id                             
    def evaluat_prediction(self):
        rootPath="/home/payson/Documents/eclipse_workspace/python3/ReinforcementInvesting/forex/predictions/"
        #rootPath="/media/payson/hdd1/tensorflow/epochs-5/"
              
        predictions=pd.DataFrame({util.states_dict[i+1]:pd.read_csv(rootPath+"fx_susd"+util.states_dict[i+1]+"predictions.csv",header=None).as_matrix().reshape(-1)
                     for i in range(util.state_dict_len-1)})
        rewards=pd.DataFrame({util.states_dict[i+1]:pd.read_csv(rootPath+"fx_susd"+util.states_dict[i+1]+"rewards.csv",header=None).as_matrix().reshape(-1)
                     for i in range(util.state_dict_len-1)})
        #print(predictions)
        #print(rewards)
        curr_ids=predictions.idxmax(axis=1).as_matrix()
        
        #curr_ids=rewards.idxmax(axis=1).as_matrix()
        real_rewards=np.array([rewards.loc[i,[curr_ids[i]]].as_matrix()[0] for i in range(len(curr_ids))])
        #real_rewards=np.array([rewards.loc[i,[self.epsilon_max_idx(curr_ids[i])]].as_matrix()[0] for i in range(len(curr_ids))])
        print(np.prod(real_rewards+1))   
         
    def soft_evaluate_prediction(self):   
        rootPath="/home/payson/Documents/eclipse_workspace/python3/ReinforcementInvesting/forex/predictions/"
        #rootPath="/media/payson/hdd1/tensorflow/epochs-5/"
              
        predictions=pd.DataFrame({util.states_dict[i+1]:pd.read_csv(rootPath+"fx_susd"+util.states_dict[i+1]+"predictions.csv",header=None).as_matrix().reshape(-1)
                     for i in range(util.state_dict_len-1)}).as_matrix()
        rewards=pd.DataFrame({util.states_dict[i+1]:pd.read_csv(rootPath+"fx_susd"+util.states_dict[i+1]+"rewards.csv",header=None).as_matrix().reshape(-1)
                     for i in range(util.state_dict_len-1)}).as_matrix()
        
        #print(rewards)
        predictions_sortidx=np.argsort(predictions)
        (r,c)=predictions_sortidx.shape
        print(predictions[2])
        print(rewards[1])
        print(predictions_sortidx[1])
        reward=1
        curr_num=1
        for i in range(r-1):
            #print(rewards[i,predictions_sortidx[i][-3:]])
            '''
            with predictions 1 step ahead of rewards, the result will be quite handsome. That means using the LSTM network to
            simulate the real data will have a delay. Moving the curve of predictions 1 step to the left will look much like the reward curve
            That means using daily price may not be possible. I should use more frequent data. Data for every hour or every minute would be better
            '''
            reward=reward*(np.sum(rewards[i,predictions_sortidx[i+1][-curr_num:]])/curr_num+1)
        print(reward)  
                                          
    def get_next_state(self):
        next_states=np.matmul(self.states,self.states_states)
        print(next_states)
        self.states=next_states  
        return self.importance_sampling(self.states)  
     
    def importance_sampling(self,state=None):   
        sample_value=np.random.random_sample()
        accumulated_value=0
        sample_index=0       
        for item in state[0]:
            accumulated_value+=item
            if accumulated_value>sample_value:
                break
            sample_index+=1

        return sample_index
    def evaluation(self):
        return
def main(_):
    with tf.Graph().as_default():
        mc=montecarlo()
        #mc.lstm_process()
        #mc.rl_process(lookforward_steps=2)
        #mc.lstm_process_all_currency()
        
        mc.lstm_test_all_currency()
        #mc.evaluat_prediction()
        #mc.soft_evaluate_prediction()
  
if __name__ =="__main__":
    tf.app.run()   
    