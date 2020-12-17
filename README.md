# kdp2df

This script loops through Kindle Direct Publishing royalty report files and creates a single Pandas dataframe that has all your KDP data. Pandas is one of the most widely used data analysis libraries for Python and comes with a dizzying array of analytics & visualization tools.

## Requirements

1. Python > 3 and < 3.9 until pandas and matplotlib run reliably on Python 3.9 and above.

2. Python packages specified in requirements.txt.

## Installation

1.  git clone this repo to a directory on your PC. 

2.  pip install -r requirements.txt

3.  **Download KDP data files** from your KDP admin page to the directory kdpdata.  You want KDP > Reports > Prior Months' Royalties.  You don't need to rename the files.

## Running the Script

**python kdp2df.py**

## About Exchange Rates

The program comes with a small static database of historical exchange rates that are used to calculate US dollar equivalent royalties for past months transactions.  The data is current through December 2020.  You can run the helper program, get_static_exchange_rates.py, to update the database.



