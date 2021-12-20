'''
  This program gets a stocks opening and closing price given the
  stock abreviation and date as input.
'''

import yfinance as yf
import sys


if len(sys.argv) != 3:
  sys.exit('usage: python3 get_stonks.py [name-of-stock] [date: yyyy-mm-dd]')

name = sys.argv[1]
date_start = sys.argv[2]

# adds 1 day to the date passed in
date_end = date_start[:(len(date_start)-2)] + str(int(date_start[len(date_start)-2:])+1)

data = yf.download(name, start=date_start, end=date_end)
price_open = (data.tail(1)['Open'].iloc[0])
price_close = (data.tail(1)['Close'].iloc[0])
diff = price_open-price_close

print(f'name: {name}')
print(f'  date: {date_start}')
print(f'    open price:  ${price_open}')
print(f'    close price: ${price_close}')
if diff < 0:
  print(f'    total gain/loss: -${diff * -1}')
else:
  print(f'    total gain/loss: ${diff}')

