# -*- coding:utf-8 -*- 
'''
Created on 2017年7月29日

@author: tanpayson
'''
import sys
sys.path.append("../db")
from dbCap import dbCap 

from urllib.request import urlopen


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

class ForexSpider():
    def __init__(self):
        self.dateklineurlstub="http://vip.stock.finance.sina.com.cn/forex/api/jsonp.php/var%20_{0}2017_7_10=/NewForexService.getDayKLine?symbol={0}&_=2017_7_10"
        self.minklineurlstub="http://vip.stock.finance.sina.com.cn/forex/api/jsonp.php/var%20_{0}_{1}_1499702171006=/NewForexService.getMinKline?symbol={0}&scale={1}&datalen=1240"
        self.db=dbCap()
        self.types=[5,15,30,60]
        #self.types=[5]
        self.pairs=['fx_seurjpy', 'fx_sgbpjpy', 'fx_seurgbp', 'fx_seurchf', 'fx_shkdusd', 'fx_seuraud', 'fx_seurcad', 'fx_sgbpaud', 'fx_sgbpcad', 'fx_schfjpy', 'fx_sgbpchf', 'fx_scadjpy', 'fx_saudjpy', 'fx_seurnzd', 'fx_sgbpnzd']
        #self.pairs=['fx_susdcad','fx_scadcny','fx_susdcny']
        self.dayKlineSql="insert into forex_day_kLine(pair,date,open,low,high,close) values ('%s','%s','%s','%s','%s','%s')"
        self.minKlineSql="insert into forex_min_kLine(pair,datetime,type,open,low,high,close) values ('%s','%s',%d,'%s','%s','%s','%s')"
    def _getCurrencyPairs(self,mainCurrency):
        pairs=[]      
        for key in currency_dict.keys():
            if key != mainCurrency:
                newPair='fx_s'+mainCurrency+key
                pairs.append(newPair)
        return pairs
    def processDayKline(self):
        for pair in self.pairs:
            print ("pair:"+pair)
            dataklineurl=self.dateklineurlstub.format(pair)
            print ("dataklineurl:"+dataklineurl)
            
            html = urlopen(dataklineurl)
            rlt=html.read().decode('gbk')
            idx=rlt.find("new String")
            forexDataList=rlt[idx+12:-4].split("|")
            for forexData in forexDataList:
                forexItems=forexData.split(",")
                if len(forexItems)>4:
                    sql=self.dayKlineSql%(pair,forexItems[0],forexItems[1],forexItems[2],forexItems[3],forexItems[4])
                    rlt=self.db.execute(sql)
                else:
                    print("forexData error:"+forexData)
            
        
    def processMinKline(self):
        for pair in self.pairs:
            print ("pair:"+pair)
            for type in self.types:
                minklineurl=self.minklineurlstub.format(pair,type)
                print ("minklineurl:"+minklineurl)
                
                
                html = urlopen(minklineurl)
                rlt=html.read().decode('gbk')
                idx=rlt.find("[")
                forexDataList=rlt[idx:-2].split("}")
                for forexData in forexDataList:
                    #print (forexData[2:])
                    forexItems=forexData[2:].split(",")
                    if len(forexItems)>4:
                        sql=self.minKlineSql%(pair,forexItems[0][3:-1],type,forexItems[1][3:-1],forexItems[2][3:-1],forexItems[3][3:-1],forexItems[4][3:-1])
                        #print(sql)
                        self.db.execute(sql)
                    else:
                        print ("forexData error:"+forexData)
            
        
    def process(self):
        self.pairs=self._getCurrencyPairs('usd')
        print(self.pairs)
        #self.processDayKline()
        self.processMinKline()
        self.db.close()
if __name__ == '__main__':
    forexSpidera= ForexSpider()
    forexSpidera.process()
