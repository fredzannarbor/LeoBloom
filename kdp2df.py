import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import argparse 
import numpy as np
from forex_python.converter import CurrencyRates
import datetime
import time

parser = argparse.ArgumentParser()
parser.add_argument("--infile", help = "seed file", default = 'select_criteria/isbns')
parser.add_argument("--outfile", help = "path to outfile", default = 'results.xlsx')
parser.add_argument("--kdppath", help = "path to kdp reports data folder", default = 'kdpdata')

args = parser.parse_args()

input_file = args.infile
output_file = args.outfile
kdppath = args.kdppath


today = pd.to_datetime("today")

historic = pd.read_csv('exchangerates/historic.csv') # historic currency rates
historic.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
historic.set_index('Date', inplace=True)
 
# now read data in KDP Prior Months Royalties report format 
#  initially supports only post-2015 format

dkdp = pd.DataFrame()

for i in glob.glob(r'kdpdata/*.xlsx'):
    print('reading data file', i)
    kdpdata = pd.read_excel(i)
    kdpdate = kdpdata.columns[1]
    kdpdata = pd.read_excel(i, header=1)
    
    # get month & date from file
    
    kdpdata['month'] = kdpdate.split()[0]
    kdpdata['year'] = kdpdate.split()[1]
    long_month_name = kdpdate.split()[0]
    year_name = kdpdate.split()[1]
    month_number = datetime.datetime.strptime(long_month_name, "%B").month
    year_number  = datetime.datetime.strptime(year_name, "%Y").year
    exchangedate = datetime.datetime(year_number, month_number, 1)
    lookupdate = exchangedate.strftime('%F')
    cusd = 1.0
    cgbp = historic.loc[lookupdate, 'GBP']
    ceur = historic.loc[lookupdate, 'EUR']
    cjpy = historic.loc[lookupdate, 'JPY']
    caud = historic.loc[lookupdate, 'AUD']
    ccad = historic.loc[lookupdate, 'CAD']
    cbrl = historic.loc[lookupdate, 'BRL']
    cmxn = historic.loc[lookupdate, 'MXN']
    cinr = historic.loc[lookupdate, 'INR']
    #cgbp = historic[historic['Date'] == lookupdate]['GBP']

    conditions = [ 
        kdpdata['Currency'] == 'USD',
        kdpdata['Currency'] == 'GBP',
        kdpdata['Currency'] == 'EUR',
        kdpdata['Currency'] == 'JPY',
        kdpdata['Currency'] == 'AUD',
        kdpdata['Currency'] == 'CAD',
        kdpdata['Currency'] == 'BRL',
        kdpdata['Currency'] == 'MXN',
        kdpdata['Currency'] == 'INR']

    choices = [ 
        kdpdata['Royalty'] * cusd,
        kdpdata['Royalty'] * cgbp,
        kdpdata['Royalty'] * ceur,
        kdpdata['Royalty'] * cjpy,
        kdpdata['Royalty'] * caud,
        kdpdata['Royalty'] * ccad,
        kdpdata['Royalty'] * cbrl,
        kdpdata['Royalty'] * cmxn,
        kdpdata['Royalty'] * cinr
        ]

    kdpdata['USDeq_Royalty'] = np.select(conditions, choices, default=0)
    kdpdata['USDeq_Royalty'] = kdpdata['USDeq_Royalty'].round(2)
    # add column with exchange rate
    dkdp = dkdp.append(kdpdata, ignore_index=True)


netunits = dkdp.groupby(['Title'], as_index=False)[['Title', 'Net Units Sold']].sum().sort_values(by='Net Units Sold', ascending = False)
print(dkdp['month'].value_counts())
print('unique ASINs with sales', dkdp['ASIN'].nunique())

print(' ')
print('Kindle Report')
print('most profitable Kindle titles')
print(dkdp.groupby('Title').sum().sort_values(by='USDeq_Royalty', ascending=False).head(10))

print('---')
print('dashboard')
print(' ')

KDP_LTDrev = dkdp['USDeq_Royalty'].sum()


print("LTD KDP DEq earnings: ", f"${KDP_LTDrev:,.0f}")
print("unique ASINs with sales: ", dkdp['ASIN'].nunique())
print("Net KDP unit sales: ", dkdp['Net Units Sold'].sum())

