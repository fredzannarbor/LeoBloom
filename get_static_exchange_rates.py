#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:52:07 2020

@author: fred
"""
import datetime
import pandas as pd
import time
from forex_python.converter import CurrencyRates

c = CurrencyRates()

colnames = [ 'GBP', 'EUR', 'JPY', 'INR', 'CAD', 'BRL', 'MXN', 'AUD']
df = pd.DataFrame(data=None, columns=colnames)

for x in range(2015, 2021):
    for y in range(1, 13):
        print(x, y)
        year_number  = x
        month_number = y
        
        exchangedate = datetime.datetime(year_number, month_number, 1)
        print(exchangedate)
        
        cgbp = c.get_rate('GBP', 'USD', exchangedate)
        ceur = c.get_rate('EUR', 'USD', exchangedate)
        cjpy = c.get_rate('JPY', 'USD', exchangedate)
        cinr = c.get_rate('INR', 'USD', exchangedate)
        ccad = c.get_rate('CAD', 'USD', exchangedate)
        cbrl = c.get_rate('BRL', 'USD', exchangedate)
        cmxn = c.get_rate('MXN', 'USD', exchangedate)
        caud = c.get_rate('AUD', 'USD', exchangedate)
        time.sleep(2)
        row = [cgbp, ceur, cjpy, cinr, ccad, cbrl, cmxn, caud]
        df.loc[exchangedate] = row

df.to_csv('historic.csv')

