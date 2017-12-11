# -*- coding:utf-8 -*- 
'''
Created on 2017年7月19日

@author: tanpayson
'''
import MySQLdb
class dbCap():
    def __init__(self):
        #don't use localhost, or else there will be an exception:Can't connect to local MySQL server through socket '/var/run/mysqld/mysqld.sock
        self.db = MySQLdb.connect("127.0.0.1","root","123456","crapy" ,charset="utf8")
        self.cursor = self.db.cursor()
        self.count=0
    def execute(self,sql):
        try:            
            self.cursor.execute(sql)
            self.db.commit()
            return 0
            #print "db execute"
        except Exception as e:
            print(e)
            print ("roll back")
            print ("sql:"+sql)
            self.db.rollback()
            return -1
    def batchExecute(self,sql):
        try:            
            self.cursor.execute(sql)
            self.count=self.count+1
            if self.count%5000 ==0:
                self.db.commit()
            #print "db execute"
        except Exception as e:
            print(e)
            print ("roll back")
            print ("sql:"+sql)
            #self.db.rollback()
    def commit(self):
        self.db.commit()
    def getCursor(self):
        return self.cursor
    def close(self):
        self.db.commit()
        self.db.close()