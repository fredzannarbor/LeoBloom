import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import argparse 
import numpy as np
from forex_python.converter import CurrencyRates
import datetime
import time
from sqlalchemy import create_engine

engine = create_engine('sqlite:///salesanalysis.db', echo=True)
sqlite_connection = engine.connect()

parser = argparse.ArgumentParser()
parser.add_argument("--infile", help = "seed file", default = 'select_criteria/isbns')
parser.add_argument("--outfile", help = "path to outfile", default = 'results.xlsx')
parser.add_argument("--lsipath", help = "path to lsi yearly folder", default = '.')

args = parser.parse_args()

input_file = args.infile
output_file = args.outfile
lsipath = args.lsipath

mrr = 1500

myfile = open(input_file, 'r')
select_criteria = myfile.readlines()
print('criteria file was ', input_file)

def create_date_variables():
    today = pd.to_datetime("today")
    thisyear = datetime.date.today().year
    starting_day_of_current_year = datetime.date.today().replace(year=thisyear, month=1, day=1)
    daysYTD = datetime.date.today() - starting_day_of_current_year
    annualizer = 365 / daysYTD.days
    
    return today, thisyear, starting_day_of_current_year, daysYTD, annualizer

def foreign_exchange_rates():

    c = CurrencyRates()
    cgbp = c.get_rate('GBP', 'USD')
    
    historic = pd.read_csv('exchangerates/historic.csv') # historic currency rates
    historic.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
    historic.set_index('Date', inplace=True)

    return historic, cgbp

def ingest_lsi():
    df = pd.DataFrame()
    for i in glob.glob(r'lsidata/*.xls'):
        
        data = pd.read_csv(i, sep='\t', lineterminator='\r', encoding='cp1252')
        yearval = os.path.splitext(i)[0]
        
        lasttwo = yearval[-2:]
        if  lasttwo == "UK" :
            yearval = yearval[:-2]
            #print('read data for UK ', yearval)
        elif  lasttwo == "GC" :
            #print(lasttwo, 'lasttwo')
            yearval = yearval[:-2]
            #print('read data from Global Connect file', lasttwo)
        else:
            pass
        data.insert(1, 'year', yearval)
        data = data[:-1] #necessary to remove extra lines
        df = df.append(data, ignore_index=True)
    
    print('read LSI data and created dataframe')
    
    sqlite_table = "lsidata"
    df.to_sql(sqlite_table, sqlite_connection, if_exists='replace')
    
    return df

def ingest_kdp(historic):
    print(historic)
    #  ingest_kdp initially supports only post-2015 format KDP data
    dkdp = pd.DataFrame()
    
    for i in glob.glob('kdpdata/*.xlsx'):
        #print("i is", i)
        kdpdata = pd.read_excel(i)
        kdpdate = kdpdata.columns[1]
            
        kdpdata = pd.read_excel(i, header=1, sheet_name=0)
        kdpdata['sheet'] = 'e-books'
        # get month & date from file
        
        kdpdata['month'] = kdpdate.split()[0]
        kdpdata['year'] = kdpdate.split()[1]
        long_month_name = kdpdate.split()[0]
        year_name = kdpdate.split()[1]
        month_number = datetime.datetime.strptime(long_month_name, "%B").month
        year_number  = datetime.datetime.strptime(year_name, "%Y").year
        exchangedate = datetime.datetime(year_number, month_number, 1)
        lookupdate = exchangedate.strftime('%F')
        print(lookupdate)
        cusd = 1.0
        cgbp = historic.loc[lookupdate, 'GBP']
        ceur = historic.loc[lookupdate, 'EUR']
        cjpy = historic.loc[lookupdate, 'JPY']
        caud = historic.loc[lookupdate, 'AUD']
        ccad = historic.loc[lookupdate, 'CAD']
        cbrl = historic.loc[lookupdate, 'BRL']
        cmxn = historic.loc[lookupdate, 'MXN']
        cinr = historic.loc[lookupdate, 'INR']

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
        
        kdpdata['Units Sold'] = kdpdata['Units Sold'] - kdpdata['Units Refunded']
        
        dkdp = dkdp.append(kdpdata, ignore_index=True)
        kdpdata = pd.read_excel(i, header=1, sheet_name=2)
        kdpdata['sheet'] = 'paperbacks'
        
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
        
        kdpdata['Units Sold'] = kdpdata['Units Sold'] - kdpdata['Units Refunded']
    
        # add column with exchange rate
    
        dkdp = dkdp.append(kdpdata, ignore_index=True)

    print('read KDP data and created dataframe dkdp')
    sqlite_table = "dkdpdata"
    dkdp.to_sql(sqlite_table, sqlite_connection, if_exists='replace')
    
    return dkdp

def ingest_books_in_print():

    pubdates = pd.read_excel('BIPtruth.xlsx', parse_dates=[18])
    pubdates = pubdates.replace({'TRUE':True,'FALSE':False})
    
    
    today = pd.to_datetime("today")
    pubdates['pub_age']= today - pubdates['Pub Date']
    pubdates['lmip'] = pubdates['pub_age'] / np.timedelta64(1, 'M')
    pubdates['lmip'] = pubdates['lmip'].round(2)
    
    subpub = pubdates[['Pub Date', 'isbn_13','ASIN', 'product_id', 'title', 'product_line', 'royaltied', 'public_domain_work', 'lmip', 'royaltied_author_id', 'US Listprice']].copy()
    
    subpub['product_id'] = subpub.product_id.astype(str)
    
    subpub.to_excel('results/subpub.xlsx')
    
    return pubdates, subpub


def ingest_direct_sales():

    directdf = pd.read_excel('directdata/LTDdirect.xlsx')
    print('read direct sales data and created dataframe directdf')
    
    return directdf

def enhance_LSI_dataframe(df, subpub, cgbp):
    df['USDeq_pub_comp'] = np.where(df['reporting_currency_code']== 'GBP', (df['YTD_pub_comp'] * cgbp).round(2), df['YTD_pub_comp'])
    
    
    df['YTD_net_quantity'] = df['YTD_net_quantity'].fillna(0.0).astype(int)
    df['isbn_13'] = df['isbn_13'].fillna(0).astype(int)
    df['YTD_net_quantity'].fillna(0.0).astype(int)
    df['YTD_pub_comp'].fillna(0.0).astype(int)
    df['USDeq_pub_comp'].fillna(0.0).astype(int)
    
    # create enhanced LSI adataframes
    print(df.info)
    enhanced = pd.merge(df, subpub, on='isbn_13', how = 'outer')
    
    pd.set_option('max_colwidth', 25)
    
    enhanced.set_index(['isbn_13'])
    
    return enhanced

def create_LTD_LSI_sales_dataframe(enhanced):
    LTD = enhanced.groupby(['title_x', 'lmip', 'author', 'isbn_13', 'product_line', 'royaltied', 'public_domain_work', 'year', 'royaltied_author_id', 'page_count', 'US Listprice'], as_index=False)[['title_x', 'YTD_net_quantity', 'USDeq_pub_comp']].sum()
    LTD['monthly_avg_$'] = (LTD['USDeq_pub_comp'] / LTD['lmip']).round(2)
    print('---')
    print('LTD top 10 by monthly averge')
    
    print(LTD.sort_values(by='monthly_avg_$', ascending=False).head(10))
    print('\n', 'LTD describe', '\n')
    print(LTD.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    LTD.to_excel('results/LTD.xlsx')
    return LTD

def create_frontlist_dataframes(LTD):

    frontlist = LTD[LTD['lmip'] < 12.0].sort_values(by='monthly_avg_$', ascending=False)
    frontlist_number = (LTD[LTD['lmip'] < 12.0]).isbn_13.size
    
    winsorized = (frontlist['USDeq_pub_comp'] > frontlist['USDeq_pub_comp'].quantile(0.05)) & (frontlist['USDeq_pub_comp'] < frontlist['USDeq_pub_comp'].quantile(0.95))
    frontlist.to_excel('results/frontlist.xlsx')
    winsorized_mean = frontlist[winsorized]['monthly_avg_$'].mean()
    
    print('---')
    print('frontlist')
    print(frontlist)
    print(frontlist.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    
    fwmean = frontlist[winsorized]['monthly_avg_$'].mean()
    
    return frontlist, frontlist_number, winsorized, winsorized_mean, fwmean

def create_product_lines_dataframe(LTD):
    print('\n', 'product lines')
    
    productlines = LTD.groupby('product_line').sum().sort_values(by='monthly_avg_$',ascending=False)
    productlines.to_excel('results/productlines.xlsx')
    print(productlines)
    return productlines

def create_by_years_dataframe(enhanced):

    pd.set_option('display.max_rows', 1000)
    pivotall = enhanced.pivot_table(index='title_x', columns='year', values='USDeq_pub_comp', aggfunc='sum', margins=True).sort_values(by='All', ascending=False).iloc[:, :-1]
    pivotall.to_excel('results/pivotall.xlsx')
    by_years = pivotall.apply(lambda x: pd.Series([e for e in x if pd.notnull(e)]), axis=1)
    by_years = by_years.drop(by_years.index[0])
    by_years.to_excel('results/by_years.xlsx')
    
    return by_years, pivotall

def create_public_domain_dataframe(enhanced, subpub):

    publicdomain = enhanced.pivot_table(index='product_id', columns='year', values='USDeq_pub_comp', aggfunc='sum', margins=True).sort_values(by='All', ascending=False)
    publicdomain = pd.merge(publicdomain, subpub, on='product_id', how='left')
    #publicdomain = publicdomain[publicdomain['public_domain_work'].fillna(False)]
    
    publicdomain['monthly_avg_$'] = (publicdomain['All'] / publicdomain['lmip']).round(2)
    colz = ['title', 'All', 'lmip', 'public_domain_work', 'monthly_avg_$']
    publicdomain[colz].sort_values(by='monthly_avg_$', ascending=False).to_excel('results/publicdomain.xlsx')
    
    print('--')
    print('public domain title performance')
    print(publicdomain[colz].sort_values(by='All', ascending=False))
    
    sqlite_table = "publicdomain"
    publicdomain.to_sql(sqlite_table, sqlite_connection, if_exists='replace')
    
    print(publicdomain[colz].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    pdbyproductline= publicdomain.groupby('product_line').sum()
    print(pdbyproductline.sort_values(by='monthly_avg_$', ascending=False))
    return publicdomain

def create_royaltied(enhanced, subpub):

    royaltied = enhanced.pivot_table(index='product_id', columns='year', values='USDeq_pub_comp', aggfunc='sum', margins=True).sort_values(by='All', ascending=False)
    royaltied = pd.merge(royaltied, subpub, on='product_id', how='left')
    #royaltied = royaltied[royaltied['royaltied'].fillna(False)]
    royaltied['monthly_avg_$'] = (royaltied['All'] / royaltied['lmip']).round(2)
    colz = ['title', 'All', 'lmip', 'royaltied', 'monthly_avg_$']
    royaltied[colz].sort_values(by='monthly_avg_$', ascending=False).to_excel('results/royaltied.xlsx')
    sqlite_table = "royaltied"
    royaltied.to_sql(sqlite_table, sqlite_connection, if_exists='replace')
    
    return royaltied

def enhance_KDP_dataframe(dkdp, subpub):

    netunits = dkdp.groupby(['Title'], as_index=False)[['Title', 'Units Sold']].sum().sort_values(by='Units Sold', ascending = False)
    
    asin2author = subpub.drop_duplicates(['ASIN'])
    dkdp['royaltied_author_id'] = dkdp['ASIN'].map(asin2author.set_index('ASIN')['royaltied_author_id'])
    enhanced_dkdp = dkdp
    
    print(' ')
    print('KDP Report')
    print('most profitable KDP titles')
    
    print(dkdp.groupby('Title').sum().sort_values(by='USDeq_Royalty', ascending=False).head(10))
    topkdptitles=dkdp.groupby('Title').sum().sort_values(by='USDeq_Royalty', ascending=False).head(10)
    return enhanced_dkdp, topkdptitles, netunits

def royalties(year, enhanced, enhanced_dkdp):
    authordata = pd.read_excel('authordata/royaltied_authors.xlsx')
    print(enhanced.info())
    print(enhanced.columns)
    yearLSIsales = enhanced[enhanced['year'].str.contains('2020').fillna(False)]
    yearKDPsales = enhanced_dkdp[enhanced_dkdp['year'].str.contains('2020').fillna(False)]
   
    #print(yearLSIsales)
    for index, row in authordata.iterrows():
        #print(row)
        yearLSIcols = ['year', 'YTD_net_quantity', 'USDeq_pub_comp', 'title_x','product_id', 'reporting_currency_code']
        yearKDPcols = ['year', 'Net Units Sold', 'USDeq_Royalty', 'Title', 'ASIN', 'royaltied_author_id']
        LSI_editions_cols = ['year', 'reporting_currency_code', 'title_x', 'product_id', 'YTD_net_quantity', 'USDeq_pub_comp']
        #LSI_editions_sales = yearLSIsales[yearLSIsales['royaltied_author_id'] == row['royaltied_author_id']][LSI_editions_cols]
        #print(yearLSIsales.columns)
        LSI_editions_sales = yearLSIsales[yearLSIsales['royaltied_author_id'] == row['royaltied_author_id']][yearLSIcols]
        LSIprofit = yearLSIsales[yearLSIsales['royaltied_author_id'] == row['royaltied_author_id']][yearLSIcols]['USDeq_pub_comp'].sum().round(2)
       # LSI_editions = yearLSIsales[yearLSIsales['royaltied_author_id'] == row['royaltied_author_id']][yearLSIcols]
        KDP_editions_cols = ['Currency','Title', 'ASIN','Net Units Sold', 'USDeq_Royalty']
        KDP_editions_sales = yearKDPsales[yearKDPsales['royaltied_author_id'] == row['royaltied_author_id']][KDP_editions_cols]
        KDPprofit = yearKDPsales[yearKDPsales['royaltied_author_id'] == row['royaltied_author_id']][yearKDPcols]['USDeq_Royalty'].sum().round(2)
        totalnetrevenue = (LSIprofit + KDPprofit).round(2)
        royaltydue = (totalnetrevenue * .30).round(2) # address escalators in future
        if not LSI_editions_sales.empty and not KDP_editions_sales.empty:
            pd.options.display.width = 72
            pd.options.display.max_colwidth = 20
            with open('royaltyreports/'+str(row['royaltied_author_id']) + str(row['Last name']) + '.txt', 'a') as outfile:
                print(year, ' royalty report - Nimble Books LLC', file=outfile)
                print(row['Real author full name'],file=outfile)
                if not LSI_editions_sales.empty:
                    print('LSI sales in ', year, 'from annual report', file=outfile)
                    print(LSI_editions_sales.to_string(index=False, max_colwidth=20), file=outfile)
                    print('LSI net revenue 2020', LSIprofit, file=outfile)
                    print('\n\n', file=outfile)
                
                if not KDP_editions_sales.empty:
                    print('KDP sales in ', year, 'from monthly reports', file=outfile)
                    print(KDP_editions_sales.to_string(index=False,  max_colwidth=20), file=outfile)
                    print('KDP net revenue 2020', KDPprofit, file=outfile)
                print('\n\n', file=outfile)
                print('total net revenue', totalnetrevenue, file=outfile)
                print('royalty due', royaltydue, file=outfile)
                print('\n' + '\n', file=outfile)
                outfile.close()
        
    return yearKDPsales, yearLSIsales, authordata, LSI_editions_sales

def jumbo(page_count, LTD):
    jumbodf = LTD[(LTD['page_count'] >= page_count) | (LTD['US Listprice'] >= 99.00)]
    jumbotitles =  jumbodf.groupby('title_x').sum().sort_values(by='monthly_avg_$',ascending=False)
    jumbotitles.index = jumbotitles.index.str[-25:]
    pd.options.display.max_colwidth = 32
    jumbotitles.info()
    print(jumbotitles.describe())
    #jumbocols =
    print(jumbotitles[['YTD_net_quantity', 'USDeq_pub_comp', 'monthly_avg_$']])
    
    return jumbodf, jumbotitles

def dashboard(pivotall, dkdp, directdf, thisyear, annualizer, goal, mrr, winsorized_mean, frontlist):
    print(pivotall.info)
    LSI_YTDrev = pivotall.iloc[0,-1].round()
    KDP_YTDrev = dkdp[dkdp['year'] == '2021']['USDeq_Royalty'].sum()
    include = directdf[directdf['Sale date'].dt.year == thisyear]
    direct_YTDrev = include['USDeq_pub_comp'].sum()
    YTD_totalrev = LSI_YTDrev + KDP_YTDrev + direct_YTDrev
    annualized_rev= YTD_totalrev * annualizer
    mrr = annualized_rev / 12
    goal = goal
    gap = goal - mrr
    gap_titles_to_do = int(gap / winsorized_mean)

    wmean = frontlist[winsorized]['monthly_avg_$'].mean()
    print('---')
    print('dashboard')
    print(' ')
    print("Goal MRR: $10,000")
    print("Current MRR: ", f"${mrr:,.0f}")
    print("Gap: ", f"${gap:,.0f}")
    print("YTD LSI DEq revenue: ", f"${LSI_YTDrev:,.0f}")
    print("YTD KDP DEq revenue: ", f"${KDP_YTDrev:,.0f}")
    print("YTD direct revenue: ", f"${direct_YTDrev:,.0f}")
    print("YTD total revenue:", f"${YTD_totalrev:,.0f}")
    print("annualized revenue:", f"${annualized_rev:,.0f}")
    
    print("unique ASINs with sales: ", dkdp['ASIN'].nunique() + dkdp['ISBN'].nunique())
    print("Net KDP unit sales: ", dkdp['Net Units Sold'].sum().astype(int))
    
    print("new LSI titles in last twelve months: ", frontlist_number)
    print("monthly avg contribution of new titles: ", frontlist['monthly_avg_$'].sum().round(2))
    
    print("Winsorized mean revenue per frontlist title: ", f"${wmean:,.2f}")
    
    print('New mean revenue public domain titles needed until goal: ', gap_titles_to_do)
    #print("Winsorized mean revenue per public domain title")
    print()
    return

if __name__ == "__main__":
    
    (today, thisyear, starting_day_of_current_year, daysYTD, annualizer) = create_date_variables()
    historic = foreign_exchange_rates()[0]
    cgbp = foreign_exchange_rates()[1]
    print(historic)
    df = ingest_lsi()
    dkdp = ingest_kdp(historic)
    (pubdates, subpub) = (ingest_books_in_print()[0], ingest_books_in_print()[1])
    directdf = ingest_direct_sales()
    enhanced = enhance_LSI_dataframe(df, subpub, cgbp)
    (dkdp, netunits) = (enhance_KDP_dataframe(dkdp, subpub)[0],[1])
    netunits = enhance_KDP_dataframe(dkdp, subpub)[1]
    LTD = create_LTD_LSI_sales_dataframe(enhanced)
    (frontlist, frontlist_number, winsorized, winsorized_mean, fwmean) = create_frontlist_dataframes(LTD)
    product_line = create_product_lines_dataframe(LTD)
    (by_years, pivotall) = (create_by_years_dataframe(enhanced)[0],create_by_years_dataframe(enhanced)[1])
    publicdomain = create_public_domain_dataframe(enhanced, subpub)
    royaltied = create_royaltied(enhanced, subpub)
    jumbo(800, LTD)
    royalties(2020, enhanced, dkdp)
    dashboard(pivotall, dkdp, directdf, thisyear, annualizer, 10000, mrr, winsorized_mean, frontlist)
