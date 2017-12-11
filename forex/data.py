'''
Created on Nov 4, 2017

@author: payson
'''
from __future__ import division, print_function, unicode_literals
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import edward
from sklearn.preprocessing import PolynomialFeatures,Normalizer


sys.path.append("../util")
import util 

class Daykline():
    def __init__(self):
        self.data_path="../data/daykline1121.csv"
        self.daykline_data=None
        self.pairs=util.getCurrencyPairs('usd')
        self.currency_paris=[]
        self.all_currencies_pair_diff1=[]
        self.all_currencies_pair_ave_diff=[]
        self.all_currencies_pair_pct_diff1=[]
        self.all_prices={}
        self.all_prices_diff={}
        self.all_cross_prices_pct_diff={}
        self.all_prices_pct_change={}
        self.all_price_pct_change_in_khb=None
        self.all_price_pct_corr={}
        self.all_price_pct_cov={}
        self.diff_lag_idx={"diff_1day":1,"diff_2days":2,"diff_3days":3,"diff_4days":4,
                           "diff_1week":5,"diff_2weeks":10,"diff_3weeks":15,
                           "diff_1month":20,"diff_2months":40,"diff_3months":60,"diff_4months":80,"diff_5months":100,"diff_6months":120}
        #self.type_idx={"open":2,"low":3,"high":4,"close":5}
        self.type_idx={"open":2,"low":3,"high":4,"close":5}
        self.cross_type_idx={"open_open_pct":2,"low_low_pct":3,"high_high_pct":4,"close_close_pct":5,"high_low_pct":6,"close_open_pct":7}
    def load_data(self):
        #csv_path=os.path.join(self.data_path,"daykline.csv")
        self.daykline_data= pd.read_csv(self.data_path)
        self.daykline_data["date"]=pd.to_datetime(self.daykline_data["date"])
        self.currency_paris=[self.daykline_data[self.daykline_data["pair"]==pair] for pair in self.pairs]

        return self.daykline_data
    def get_days_ave_pct(self,price,num_days):
        days_aves=[]
        for i in range(len(price)):
            if i>num_days-1:
                days_ave=np.mean(price[i-num_days+1:i+1])
                days_aves.append(days_ave)
            else:
                days_ave=np.mean(price[0:i+1])
                days_aves.append(days_ave)
        days_aves=np.array(days_aves)[0:-1]
        ave_diff=price[1:]-days_aves
        ave_diff_pct=ave_diff/days_aves     
        return ave_diff_pct
                   
    def get_oneweek_ave_pct(self,price): 
        return self.get_daysave_pct(price,5)     
    def get_twoweek_ave_pct(self,price):   
        return self.get_daysave_pct(price,10)   
    def get_threeweek_ave_pct(self,price):     
        return self.get_daysave_pct(price,15)   
    def get_onemonth_ave_pct(self,price): 
        return self.get_daysave_pct(price,20)       
    def get_twomonth_ave_pct(self,price): 
        return self.get_daysave_pct(price,40)      
    
    def get_threemonth_ave_pct(self,price): 
        return self.get_daysave_pct(price,60)   
    def get_sixmonth_ave_pct(self,price): 
        return self.get_daysave_pct(price,80) 
     
    def get_type_currencies_pair_diff(self,item,currency_pair_name,currencies_pair_ave_diff):
        (type,price)=item
        lag_items=self.diff_lag_idx.items()
        for lag_item ,lag in lag_items:
            lag_item_name=currency_pair_name+"_"+type+"_"+lag_item  
            currencies_pair_ave_diff[lag_item_name]=self.get_days_ave_pct(price,lag)
            
        return  currencies_pair_ave_diff            
    def get_all_currencies_pair_diff(self):
        for currency_pair in self.currency_paris:
            pair=currency_pair.iloc[1:,:1].as_matrix().reshape(-1)
            date=currency_pair.iloc[1:,1:2].as_matrix().reshape(-1)
            currency_pair_name=pair[0]

            open=currency_pair.iloc[:,2:3].as_matrix().reshape(-1)
            low=currency_pair.iloc[:,3:4].as_matrix().reshape(-1)
            high=currency_pair.iloc[:,4:5].as_matrix().reshape(-1)
            close=currency_pair.iloc[:,5:6].as_matrix().reshape(-1)
            
            open_open=open[1:]-open[:-1]
            open_open_pct=open_open/open[:-1]
                            
            low_low=low[1:]-low[:-1]   
            low_low_pct=low_low/low[:-1]      
            high_high=high[1:]-high[:-1]  
            high_high_pct=high_high/high[:-1]   
            close_close=close[1:]-close[:-1] 
            close_close_pct=close_close/close[:-1]        
            high_low=high[1:]-low[:-1]
            high_low_pct=high_low/low[:-1]
            close_open=close[1:]-open[:-1]
            close_open_pct=close_open/open[:-1]
            
            
            currencies_pair_diff1=pd.DataFrame({"pair":pair,
                                                "date":date,
                                                "open_open":open_open,
                                                "low_low":low_low,
                                                "high_high":high_high,
                                                "close_close":close_close,
                                                "high_low":high_low,
                                                "close_open":close_open})
            currencies_pair_diff1=currencies_pair_diff1[["pair","date","open_open","low_low","high_high","close_close","high_low","close_open"]]
            
            currencies_pair_pct_diff1=pd.DataFrame({"pair":pair,
                                                    "date":date,
                                                    "open_open_pct":open_open_pct,
                                                    "low_low_pct":low_low_pct,
                                                    "high_high_pct":high_high_pct,
                                                    "close_close_pct":close_close_pct,
                                                    "high_low_pct":high_low_pct,
                                                    "close_open_pct":close_open_pct})
            
            currencies_pair_pct_diff1=currencies_pair_pct_diff1[["pair","date","open_open_pct","low_low_pct","high_high_pct","close_close_pct","high_low_pct","close_open_pct"]]
            
            
            currencies_pair_ave_diff=pd.DataFrame({"date":date})
            
            currencies_pair_ave_diff=self.get_type_currencies_pair_diff(("open",open),currency_pair_name,currencies_pair_ave_diff)
            
            currencies_pair_ave_diff=self.get_type_currencies_pair_diff(("low",low),currency_pair_name,currencies_pair_ave_diff)
            currencies_pair_ave_diff=self.get_type_currencies_pair_diff(("high",high),currency_pair_name,currencies_pair_ave_diff)
            currencies_pair_ave_diff=self.get_type_currencies_pair_diff(("close",close),currency_pair_name,currencies_pair_ave_diff)
            
            
            self.all_currencies_pair_ave_diff.append(currencies_pair_ave_diff)            
            self.all_currencies_pair_diff1.append(currencies_pair_diff1)
            self.all_currencies_pair_pct_diff1.append(currencies_pair_pct_diff1)
            #print(currency_paris_diff1)

    def get_currency_paris(self,pairs=None):
        if pairs==None:
            self.currency_pair=[self.daykline_data[self.daykline_data["pair"]==pair] for pair in self.pairs]
        else:
            self.currency_pair=[self.daykline_data[self.daykline_data["pair"]==pair] for pair in pairs]   
            
    def get_price(self,type="close"):
        pair=self.currency_paris[0].iloc[0,0]
        index=[False,True,False,False,False,False]
        index[self.type_idx[type]]=True
        price=self.currency_paris[0].iloc[:,index]
        price_frame=price.rename(columns={type:pair+"_"+type}).set_index("date")
        #print(open_price)
        for i in range(len(self.currency_paris)-1):
            next_pair=self.currency_paris[i+1].iloc[0,0]
            next_price=self.currency_paris[i+1].iloc[:,index]
            next_price=next_price.rename(columns={type:next_pair+"_"+type})
            #print(next_open_price)
            price_frame=price_frame.join(next_price.set_index("date"),how="outer")
            #print(open_price_frame)
        length=len(price_frame.index)
        price_frame["fx_susdusd"+"_"+type]=pd.Series(np.array([1.0]*length,np.float64),index=price_frame.index)  
        currency_pair_order=[util.getCurrencyPair(util.states_dict[i])+"_"+type for i in range(len(util.states_dict))]   
        price_frame=price_frame[currency_pair_order]
        price_frame.fillna(0.0)   
        return price_frame   

    def get_all_price(self):            
        items=self.type_idx.items()
        for type,_ in items:
            price=self.get_price(type)
            self.all_prices[type]=price
        return self.all_prices
    def get_all_price_diff(self,diff=1):    
        items=self.type_idx.items()
        if len(self.all_prices)==0:
            self.get_all_price()
        for type,_ in items:
            price=self.all_prices[type]
            price_diff=price.diff(diff,axis=0)
            self.all_prices_diff[type]=price_diff
        return self.all_prices_diff
       
    def get_all_price_pct_change(self):    
        items=self.type_idx.items()
        if len(self.all_prices)==0:
            self.get_all_price()
        for type,_ in items:
            price=self.all_prices[type]
            price_pct_change=price.pct_change()
            self.all_prices_pct_change[type]=price_pct_change  
                     
        return self.all_prices_pct_change 
    def get_all_currency_pair_ave_pct_diff_pivot(self):
        num=len(self.all_currencies_pair_ave_diff)     
        if num==0:
            self.get_all_currencies_pair_diff()
            num=len(self.all_currencies_pair_ave_diff)     
        
        currencies_pair_ave_diff=self.all_currencies_pair_ave_diff[0]
        currencies_pair_ave_diff=currencies_pair_ave_diff.set_index("date")  
        #print(currencies_pair_ave_diff)
        for i in range(num-1):
            next_currencies_pair_ave_diff=self.all_currencies_pair_ave_diff[i+1]
            currencies_pair_ave_diff=currencies_pair_ave_diff.join(next_currencies_pair_ave_diff.set_index("date") ,how="outer").fillna(0)

        return currencies_pair_ave_diff
        
    def get_all_cross_prices_pct_diff(self):
        items=self.cross_type_idx.items()
        if len(self.all_currencies_pair_pct_diff1)==0:
            self.get_all_currencies_pair_diff()
        for type,_ in items:
            pair=self.all_currencies_pair_pct_diff1[0].iloc[0,0]
            index=[False]*(len(self.cross_type_idx)+2)
            index[1]=True           
            index[self.cross_type_idx[type]]=True
            #print(index)
            cross_pct_diff=self.all_currencies_pair_pct_diff1[0].iloc[:,index]
            cross_pct_diff_frame=cross_pct_diff.rename(columns={type:pair+"_"+type}).set_index("date")  
            for i in range(len(self.all_currencies_pair_pct_diff1)-1):
                next_pair=self.all_currencies_pair_pct_diff1[i+1].iloc[0,0]
                next_cross_pct_diff=self.all_currencies_pair_pct_diff1[i+1].iloc[:,index]
                #print(type)
                next_cross_pct_diff=next_cross_pct_diff.rename(columns={type:next_pair+"_"+type})
                #print(next_cross_pct_diff)
                cross_pct_diff_frame=cross_pct_diff_frame.join(next_cross_pct_diff.set_index("date"),how="outer") 
            length=len(cross_pct_diff_frame.index)
            cross_pct_diff_frame["fx_susdusd"+"_"+type]=pd.Series(np.array([0.0]*length,np.float64),index=cross_pct_diff_frame.index)   
            cross_pct_diff_frame=cross_pct_diff_frame.fillna(0.0)  
            currency_pair_order=[util.getCurrencyPair(util.states_dict[i])+"_"+type for i in range(len(util.states_dict))]      
            cross_pct_diff_frame=cross_pct_diff_frame[currency_pair_order]
            #print(cross_pct_diff_frame)            
            self.all_cross_prices_pct_diff[type]=cross_pct_diff_frame
        return self.all_cross_prices_pct_diff
    
    def get_all_cross_price_pct_diff_in_khb(self):    
        '''builing price percentage change in kernel hilbert space'''
        items=self.cross_type_idx.items()
        if len(self.all_cross_prices_pct_diff)==0:
            self.get_all_cross_prices_pct_diff()   
            
        price_cross_pct_diff_open= self.all_cross_prices_pct_diff["open_open_pct"]
        #delete the new add column to produce less o in reproducing kernel hilbert space
        del price_cross_pct_diff_open["fx_susdusd_open_open_pct"]

        
        prices_cross_pct_diff=price_cross_pct_diff_open
        
        for type,_ in items:
            if type == "open_open_pct":
                continue
            price_cross_pct_change=self.all_cross_prices_pct_diff[type]
            #delete the new add column to produce less o in reproducing kernel hilbert space
            del price_cross_pct_change["fx_susdusd_"+type]
            #print(price_cross_pct_change)
            prices_cross_pct_diff=prices_cross_pct_diff.join(price_cross_pct_change)
            
            
        #print(prices_cross_pct_diff)            
        index=prices_cross_pct_diff.index
        
        price_pct_change=prices_cross_pct_diff.fillna(0)
        #print(price_pct_change)
        price_pct_change_array=price_pct_change.as_matrix()
        price_pct_change_array=price_pct_change_array*100
        (r,c)=price_pct_change_array.shape
        #print(r,",",c)
        prices_pct_diff_in_khb=[]
        for i in range(r):
            poly = PolynomialFeatures(2)
            prices_pct_diff_in_khb_item=poly.fit_transform(price_pct_change_array[i].reshape(1,c))
            prices_pct_diff_in_khb_item=prices_pct_diff_in_khb_item.reshape(1,-1)
            #print(price_pct_change_khb)
            prices_pct_diff_in_khb.append(prices_pct_diff_in_khb_item[0])
            #price_pct_change_khb=np.matmul(price_pct_change_array,price_pct_change_array)
        prices_pct_diff_in_khb=np.array(prices_pct_diff_in_khb)
        prices_pct_diff_in_khb=pd.DataFrame(data=prices_pct_diff_in_khb,index=index)
        print(prices_pct_diff_in_khb.shape)
        #self.all_prices_pct_diff_in_khb=prices_pct_diff_in_khb  
        return prices_pct_diff_in_khb            
    
    
    def get_all_diff_pct_price_in_khb(self):
        items=self.type_idx.items()
        for type,_ in items:
            lag_items=self.diff_lag_idx.items()
            for lag_item ,_ in lag_items:
                lag_item_name=items+"_"+lag_item
    def get_all_price_pct_change_in_khb(self):    
        '''builing price percentage change in kernel hilbert space'''
        items=self.type_idx.items()
        if len(self.all_prices_pct_change)==0:
            self.get_all_price_pct_change()
   

        price_pct_change_open= self.all_prices_pct_change["open"]
        #delete the new add column to produce less o in reproducing kernel hilbert space
        del price_pct_change_open["fx_susdusd_open"]
        price_pct_change=price_pct_change_open
        
        price_pct_change_low= self.all_prices_pct_change["low"]
        #delete the new add column to produce less o in reproducing kernel hilbert space
        del price_pct_change_low["fx_susdusd_low"]
        price_pct_change=price_pct_change.join(price_pct_change_low)
        
        price_pct_change_high= self.all_prices_pct_change["high"]
        #delete the new add column to produce less o in reproducing kernel hilbert space
        del price_pct_change_high["fx_susdusd_high"]
        price_pct_change=price_pct_change.join(price_pct_change_high)
  
        price_pct_change_close= self.all_prices_pct_change["close"]
        #delete the new add column to produce less o in reproducing kernel hilbert space
        del price_pct_change_close["fx_susdusd_close"]
        price_pct_change=price_pct_change.join(price_pct_change_close)
                     
        index=price_pct_change.index
        
        price_pct_change=price_pct_change.fillna(0)
        price_pct_change_array=price_pct_change.as_matrix()
        price_pct_change_array=price_pct_change_array*10000
        (r,c)=price_pct_change_array.shape
        #print(r,",",c)
        price_pct_change_in_khb=[]
        for i in range(r):
            poly = PolynomialFeatures(2)
            price_pct_change_khb=poly.fit_transform(price_pct_change_array[i].reshape(1,c))
            price_pct_change_khb=price_pct_change_khb.reshape(1,-1)
            #print(price_pct_change_khb)
            price_pct_change_in_khb.append(price_pct_change_khb[0])
            #price_pct_change_khb=np.matmul(price_pct_change_array,price_pct_change_array)
        price_pct_change_in_khb=np.array(price_pct_change_in_khb)
        price_pct_change_in_khb=pd.DataFrame(data=price_pct_change_in_khb,index=index)
        print(price_pct_change_in_khb.shape)
        self.all_price_pct_change_in_khb=price_pct_change_in_khb  
        return price_pct_change_in_khb
    def save_all_price_pct_change_in_khb_to_csv(self):
        csv_path=os.path.join(self.data_path,"dayklineinhkb.csv")
        self.all_price_pct_change_in_khb.to_csv(csv_path)
    def get_price_pct_change_in_khb_dict(self):    
        '''builing price percentage change in kernel hilbert space'''
        items=self.type_idx.items()
        if len(self.all_prices_pct_change)==0:
            self.get_all_price_pct_change()
   
        for type,_ in items:
            price_pct_change= self.all_prices_pct_change[type]
            #delete the new add column to produce less o in reproducing kernel hilbert space
            del price_pct_change["fx_susdusd"+"_"+type]
            index=price_pct_change.index
            price_pct_change=price_pct_change.fillna(0)
            price_pct_change_array=price_pct_change.as_matrix()
            price_pct_change_array=price_pct_change_array*1000
            (r,c)=price_pct_change_array.shape
            print(r,",",c)
            price_pct_change_in_khb=[]
            for i in range(r):
                poly = PolynomialFeatures(2)
                price_pct_change_khb=poly.fit_transform(price_pct_change_array[i].reshape(1,c))
                price_pct_change_khb=price_pct_change_khb.reshape(1,-1)
                #print(price_pct_change_khb)
                price_pct_change_in_khb.append(price_pct_change_khb[0])
                #price_pct_change_khb=np.matmul(price_pct_change_array,price_pct_change_array)
            price_pct_change_in_khb=np.array(price_pct_change_in_khb)
            print(price_pct_change_in_khb)
            price_pct_change_in_khb=pd.DataFrame(data=price_pct_change_in_khb,index=index)
            self.all_price_pct_change_in_khb[type]=price_pct_change_in_khb
            print(price_pct_change_in_khb)
        
    def get_all_price_pct_corr(self):  
        items=self.type_idx.items()
        if len(self.all_price_pct_change)==0:
            self.get_all_price_pct_change()
        for type,_ in items:        
            self.all_price_pct_corr[type]=self.all_price_pct_change[type].corr() 
        return self.all_price_pct_corr     
    def get_all_price_pct_cov(self):  
        items=self.type_idx.items()
        if len(self.all_price_pct_change)==0:
            self.get_all_price_pct_change()
        for type,_ in items:        
            self.all_price_pct_cov[type]=self.all_price_pct_change[type].corr() 
        return self.all_price_pct_cov

if __name__ == '__main__':
    daykline=Daykline()
    daykline_data=daykline.load_data()
    currency_paris_diff=daykline.get_all_currencies_pair_diff()
    daykline.get_all_currency_pair_ave_pct_diff_pivot()
    #all_cross_prices_pct_diff=daykline.get_all_cross_prices_pct_diff()
    #print(all_cross_prices_pct_diff["close_close_pct"])
    #daykline.get_all_cross_price_pct_diff_in_khb()
    #print(currency_paris_diff)
    '''
    daykline.get_price()
    print(daykline.get_all_price()["close"]["2015-10":]['fx_susdcad_close'])
    
    all_price_diff=daykline.get_all_price_diff()
    print(all_price_diff["close"]["2015-10":]['fx_susdcad_close'])
    #print(all_price_diff["close"]["2015-10":]['fx_susdusd_close'].describe())
    #print(all_price_diff["close"]["2015-10":]['fx_susdhkd_close'].describe())
    #print(all_price_diff["close"]["2015-10":]['fx_susdusd_close'].cov(all_price_diff["close"]["2015-10":]['fx_susdhkd_close']))
    
    all_price_pct_change=daykline.get_all_price_pct_change()
    print("=======")
    fx_susdcad_close=all_price_pct_change['close']["2015-10":]['fx_susdcad_close'].fillna(0)
    print(fx_susdcad_close)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_title('The percent change of usd-cad')
    fx_susdcad_close.plot(ax=ax,style='k-')
    plt.show()
    print("=======")
    #print(all_price_pct_change["close"])
    currency_pair_corr=all_price_pct_change["close"].corr()
    currency_pair_cov=all_price_pct_change["close"].cov()
    print(currency_pair_corr)
    print(currency_pair_cov)
    '''
    #daykline.get_all_price_pct_change_in_khb()
    #daykline.save_all_price_pct_change_in_khb_to_csv()
    #print(currency_paris_diff[1].iloc[-100:,5])
    
    
    #fx_susdthb.plot(kind="scatter",x="open",y="close",alpha=0.1)
    #plt.legend()
    
    #print(fx_susdthb)
    

    
    