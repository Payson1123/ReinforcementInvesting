# -*- coding:utf-8 -*- 
'''
Created on Oct 1, 2017

@author: payson
'''
import sys

sys.path.append("../db")
from dbCap import dbCap 
import json
from urllib.request import urlopen
import time
import datetime
from bs4 import BeautifulSoup
import requests
from selenium import webdriver

class UsStockSpider1():
    def __init__(self):
        self.db=dbCap()
        self.tblIdx=1
        self.stocklisturlstub="http://stock.finance.sina.com.cn/usstock/api/jsonp.php/IO.XSRV2.CallbackList['fa8Vo3U4TzVRdsLs']/US_CategoryService.getList?page={0}&num=20&sort=&asc=0&market=&id="
        self.dateklineurlstub="http://stock.finance.sina.com.cn/usstock/api/jsonp_v2.php/var%20_{0}{1}=/US_MinKService.getDailyK?symbol={0}&_={1}&___qn=3"
        self.usstockcompanySqlStub="insert into us_stock_company(symbol,name,cname,category,tbl_idx) values('%s','%s','%s','%s',%d)"
        self.dayKlineSqlStub="insert into us_stock_day_kline%d(symbol,kdate,open,high,low,close,volumn) values('%s','%s',%f,%f,%f,%f,%d)"
    def storeUSStockCompany(self,company):
        symbol=company["symbol"]
        name=company["name"]
        cname=company["cname"]
        category=company["category"]
        sql=self.usstockcompanySqlStub%(symbol,name,cname,category,self.tblIdx)
        self.db.execute(sql)
    def storeUSStockHistory(self,symbol,stock):
        date=stock["d"]
        openPrice=float(stock["o"])
        highPrice=float(stock["h"])    
        lowPrice=float(stock["l"])
        closePrice=float(stock["c"])
        volumn=int(stock["v"])
        #print(stock["d"]+","+stock["o"]+","+stock["h"]+","+stock["l"]+","+stock["c"]+","+stock["v"])
        sql=self.dayKlineSqlStub%(self.tblIdx,symbol,date,openPrice,highPrice,lowPrice,closePrice,volumn)
        self.db.batchExecute(sql)
        #self.db.execute(sql)
    def getCurrDate(self):
        return time.strftime("%Y_%m_%d")
    def processStockDetail(self,storeProfile=True):
        pageIdx=406
        currDate=self.getCurrDate()
        for i in range(pageIdx):
            self.tblIdx=i/20+1
            print("page index is:"+str(i+1))
            stocklisturl=self.stocklisturlstub.format(i+1)
            #print(stocklisturl)
            html = urlopen(stocklisturl)
            rlt=html.read().decode('gbk')
            contents=rlt[43:-3].replace('{','{\"').replace(':','":').replace('null','"null"').replace('",','","').replace('"{','{').replace('\\','\\\\')
            #print(contents[:2512])
            stockprofile=json.loads(contents)
            
            for company in stockprofile["data"]:
                if storeProfile:
                    self.storeUSStockCompany(company)
                
                symbol=company["symbol"]
                dateklineurl=self.dateklineurlstub.format(symbol,currDate)
                print(dateklineurl)
                html = urlopen(dateklineurl)
                rlt=html.read().decode('gbk')
                idx=rlt.find("[")
                contents=rlt[idx:-2]
                try:
                    stockhistory=json.loads(contents)
                    for stock in stockhistory:                   
                        self.storeUSStockHistory(symbol,stock)
                except Exception as e:
                    print(e)
                
            #s=json.loads(rlt[43:-3])
            #s=json.loads('{count:"8115",data:[{name:"Apple Inc.",cname:"苹果公司",category:"计算机",symbol:"AAPL",price:"154.12",diff:"0.84",chg:"0.55",preclose:"153.28",open:"153.21",high:"154.13",low:"152.00",amplitude:"1.39%",volume:"26299810",mktcap:"796065234709",pe:"18.50180030",market:"NASDAQ",category_id:"5"}]')
            #s=json.loads('{"count":"8115",data:"APPL"}')
    def process(self):
        self.processStockDetail(True)  
        self.db.close()
        
class UsStockSpider2():
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36"}
        self.session=requests.Session()         
        self.db=dbCap()
        self.db2=dbCap()
        self.tblIdx=1
        self.period1="655099200"
        self.period2="1507089600"
        #self.dateklineurlstub="https://ca.finance.yahoo.com/quote/{0}/history?period1={1}&period2={2}&interval=1d&filter=history&frequency=1d"       
        #self.dateklineurlstub="file:///home/payson/Downloads/AAPL.html"
        self.dateklineurlstub="https://query1.finance.yahoo.com/v7/finance/download/{0}?period1={1}&period2={2}&interval=1d&events=history&crumb={3}"
        self.usstockcompanySelectSqlStub="select symbol,tbl_idx from us_stock_company"
        self.dayKlineSqlStub="insert into us_stock_day_kline%d(symbol,kdate,open,high,low,close,adjclose,volumn) values('%s','%s',%f,%f,%f,%f,%f,%s)"
    def storeUSStockHistory(self,symbol,component):
        try:
            open=round(float(component[1]),2)
        except:
            open=0
        try:
            high=round(float(component[2]),2)
        except:
            high=0
        try:
            low=round(float(component[3]),2)
        except:
            low=0
        try:
            close=round(float(component[4]),2)
        except:
            close=0
        try:
            adjclose=round(float(component[5]),2)
        except:
            adjclose=0
        sql=self.dayKlineSqlStub%(self.tblIdx,symbol,component[0],open,high,low,close,adjclose,component[6])
        self.db.batchExecute(sql)
        #self.db.execute(sql)
    def getCurrDate(self):
        return time.strftime("%Y_%m_%d")
    def getCurrDateUnix(self):
        currDate=self.getCurrDate()
        return str(int(time.mktime(datetime.datetime.strptime(currDate, "%Y_%m_%d").timetuple())))
    def processStockDetail(self,storeProfile=True):
        cursor=self.db2.getCursor()
        cursor.execute(self.usstockcompanySelectSqlStub)
        item=cursor.fetchone()
        s=requests.Session()
        scrum="u0gAnjuyln7"
        cookies=dict(B='a2rhq41ct2gln&b=3&s=v0')
        while item:
            symbol=item[0]
            self.tblIdx=item[1]
            print(symbol+","+str(self.tblIdx))
            dateklineurl=self.dateklineurlstub.format(symbol,self.period1,self.getCurrDateUnix(),scrum)
            r=s.get(dateklineurl,cookies=cookies,verify=False)
            f=open("/media/payson/hdd1/data/usstock/"+symbol+".csv","w")
            f.write(r.text)
            f.close()
            rows=r.text.split('\n')[1:]
            
            for row in rows:
                component=row.split(',')
                if len(component)==7:
                    self.storeUSStockHistory(symbol, component)
                    #self.db.batchExecute(sql)
            self.db.commit()            
            #self.storeUSStockHistory(symbol, stock)
            
            item=cursor.fetchone()
    def process(self):
        self.processStockDetail(True)  
        self.db.close()
        self.db2.close()
if __name__ == '__main__':
    print (datetime.datetime.strptime("Feb 27, 2014", "%b %d, %Y").strftime("%Y-%m-%d"))
    #usStockSpider1= UsStockSpider1()
    #usStockSpider1.process()        
    #print(len("IO.XSRV2.CallbackList['fa8Vo3U4TzVRdsLs'](("))
    usStockSpider2= UsStockSpider2()
    usStockSpider2.process()     

    