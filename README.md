# LeoBloom.py

This script loops through Kindle Direct Publishing royalty and Lightning Source, Inc. (LSI) sales report files and turns them into Pandas dataframes that are then used to produce a wide variety of reports.

Pandas is one of the most widely used data analysis libraries for Python and comes with a dizzying array of analytics & visualization tools.

Leo Bloom is the name of the accountant played by Gene Wilder in "The Producers", whom Zero Mostel enlists in his insane scheme to make an unprofitable theatre production.

## Requirements

1. Python > 3 and < 3.9 until pandas and matplotlib run reliably on Python 3.9 and above.

2. Python packages specified in requirements.txt.

## Installation

1.  git clone this repo to a directory on your PC. 

2.  pip install -r requirements.txt

## Setup

1.  **Download KDP data files** from your KDP admin page to the directory /kdpdata.  You want KDP > Reports > Prior Months' Royalties.  You don't need to rename the files.
2.  **Download LSI data files** from LSI's mail delivery option to the directory /lsidata.
3.  **Enter any direct sales in the spreadsheet** in the directory /directdata.
4.  **Customize the file authordata/royaltied_authors.xlsx** with the information about your authors.
5.  **Customize the file BIPtruth.csv** with the metadata for your list of titles.'

## Running the Script

**python LeoBloom.py**

A long series of reports will scroll across the console. In addition, various specialized reports are generated and stored in /reports.

The various dataframes generated during creation of the program are saved in /results.  These can be very handy for troubleshooting and double-checking.

## About Exchange Rates

The program comes with a small static database of historical exchange rates that are used to calculate US dollar equivalent royalties for past months transactions.  The data is current through December 2020.  You can run the helper program, get_static_exchange_rates.py, to update the database.



