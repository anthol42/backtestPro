# Calculations for Integrative test 2 - ComplexGoodStrategy
Commission: 10$ per trade
Initial cash: $100 000
Margin interest: 10%
Short interest: 20%
## Calculate trades profits
### NVDA -- Long
- Buy 200 shares on 50% margin at 440.1613
  - Total: $44 026.13
  - Debt: $44 016.13
  - Shares owned: 200
  - Average price: 440.1613
- Sell 100 shares at $483.90
  - Total: $27 371.935
  - Debt: $22 008.065
  - Shares owned: 100
  - Profit: $4353.875; 19.77%
- Buy 100 shares on 75% margin at $484.62
  - Total: $36 356.50
  - Debt: $34 123.565
  - Shares owned: 200
  - Average price: 462.39065
- Sell 200 shares at $683.19
  - Total: $102 514.435
  - Debt: $0.00000
  - Shares owned: 0
  - Profit: $44 695.6063; 77.31%

### AAPL -- Short
- Sell 200 shares at $194.8414
  - Total: $38 958.28
  - Debt: $0.00000
  - Shares due: 200
  - Average price: 194.8414
- Buy 100 shares at $181.758
  - Total: $18 185.80
  - Debt: $0.00000
  - Shares due: 100
  - Profit: $1288.34; 6.6139%
- Sell 100 shares at $195.1709
  - Total: $19 507.09
  - Debt: $0.00000
  - Shares due: 200
  - Average price: 195.0062$
- Buy 200 shares at $179.6308
  - Total: $35 936.16
  - Debt: $0.00000
  - Shares due: 0
  - Profit: 3 055.08; 7.83%

## Calculate margin interests
### NVDA
1. November: 44 016.13 * 28d * 10% / 365d = 337.6580
2. December: 44 016.13 * 13d * 10% / 365d + 22 008.065 * 16d * 10% / 365d = 253.2434 # 16 days because the last trading days in december was the 29th.  The remaining 3 days are charged in January
2. January: 22 008.065 * 6d * 10% / 365d + 34 123.565 * 27d * 10% / 365d = 288.5985
3. February: 34 123.565 * 6d * 10% / 365d = 56.0926

### AAPL
Ain't now way  calculate this by hand : The value was found from implementation  by running the backtest
(Anyway, it was already tested in unittests)
1. November: 0
2. December: 426.3514
3. January: 422.0472
4. February: 20.4526

## Calculate dividends
### NVDA
December dividend payout: 33d/90d * 200 stocks * 0.04$ = 2.93$

### Calculate account worth at every changed moment
Note: 100$ is deposit every week in the account
2 November: 100 000$
3 November: 100 000 - 44 026.12972 = 55 973.87028
6 November: 55 973.87028 + 100 = 56 073.87028
13 November: 56 073.87028 + 100 = 56 173.87028
20 November: 56 173.87028 + 100 = 56 273.87028
27 November: 56 273.87028 + 100 = 56 373.87028
1 December: 56 373.87028 - 325.6727504 = 56 048.1975296
4 December: 56 048.1975296 + 100 = 56 148.1975296
5 December: 56 148.1975296 + 3.4666666666667$ = 56 151.6642    # Dividends
11 December: 56 151.6642 + 100 = 56 251.6642
14 December: 56 251.6642 + 27 657.56474 + 39 543.52941 = 123 452.7584
18 December: 123 452.7584 + 100 = 123 552.7584
27 December: 123 552.7584 + 100 = 123 652.7584
2 January: 123 652.7584 - 271.3679229 + 100 = 123 481.3904
5 January: 123 481.3904 - 18 185.80772 - 36 356.49963 = 69 939.08305
8 January: 69 939.08305 + 100 = 70 039.08305
15 January: 70 039.08305 + 100 = 70 139.08305
22 January: 70 139.08305 + 100 = 70 239.08305
24 January: 70 239.08305 + 19 467.14746 = 89 706.23051
29 January: 89 706.23051 + 100 = 89 806.23051
2 February: 89 806.23051 - 35 936.15747 = 53 870.07304
5 February: 53 870.07304 + 100 = 53 970.07304
7 February: 53 970.07304 + 100 271.5009 = 154 241.573
End: 154 241.573 - 65.44245293 = 154 176.1305
