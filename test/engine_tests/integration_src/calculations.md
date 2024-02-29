# Calculations for Integrative test 2 - ComplexGoodStrategy
Commission: 10$ per trade
Initial cash: $100 000
Margin interest: 10%
Short interest: 20%
## Calculate trades profits
### NVDA -- Long
- Buy 200 shares on 50% margin at 440.1612972004770
  - Total: $44 026.12972
  - Debt: $44 016.12972
  - Shares owned: 200
  - Average price: 440.1612972004770
- Sell 100 shares at $496.75629603239900
  - Total: $27 657.56474
  - Debt: $22 008.06486
  - Shares owned: 100
  - Profit: $5 649.49988; 25.6701346%
- Buy 100 shares on 75% margin at $484.6199951171880
  - Total: $36 356.49963
  - Debt: $34 123.56474
  - Shares owned: 200
  - Average price: 462.3906462
- Sell 200 shares at $683.1900024414060
  - Total: $100 271.5009
  - Debt: $0.00000
  - Shares owned: 0
  - Profit: $42 472.67013; 73.4836147%

### AAPL -- Short
- Sell 200 shares at $197.7676470402110
  - Total: $39 543.52941
  - Debt: $0.00000
  - Shares due: 200
  - Average price: 197.7676470402110
- Buy 100 shares at $181.75807721578200
  - Total: $18 185.80772
  - Debt: $0.00000
  - Shares due: 100
  - Profit: $1 585.956985; 8.0213223%
- Sell 100 shares at $194.77147463800900
  - Total: $19 467.14746
  - Debt: $0.00000
  - Shares due: 200
  - Average price: 196.2695608
- Buy 200 shares at $179.6307873390830
  - Total: $35 936.15747
  - Debt: $0.00000
  - Shares due: 0
  - Profit: 3 331.75468; 8.4520357%

## Calculate margin interests
### NVDA
1. November: 44 026.12972 * 27d * 10% / 365d = 325.6727504
2. December: 44 026.12972 * 13d * 10% / 365d + 22 008.06486 * 19d * 10% / 365d = 271.3679229
2. January: 22 008.06486 * 3d * 10% / 365d + 34 123.56474 * 26d * 10% / 365d = 261.1607884
3. February: 34 123.56474 * 7d * 10% / 365d = 65.44245293

### AAPL
Ain't now way  calculate this by hand : The value was found from implementation 
(Anyway, it was already tested in unittests)

## Calculate dividends
### NVDA
December dividend payout: 39d/90d * 200 stocks * 0.04$ = 3.4666666666667$

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
