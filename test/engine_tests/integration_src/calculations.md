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
  - Total: $26 371.935
  - Debt: $22 008.065
  - Shares owned: 100
  - Profit: $4353.875; 19.77%
- Buy 100 shares on 75% margin at $484.62
  - Total: $36 356.50
  - Debt: $34 123.565
  - Shares owned: 200
  - Average price: 462.39065
- Sell 200 shares at $683.19
  - Total: $102 504.435
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
3 November: 100 000 - 44 026.13 = 55 973.87  
6 November: 55 973.87+ 100 = 56 073.87  
13 November: 56 073.87 + 100 = 56 173.87  
20 November: 56 173.87 + 100 = 56 273.87  
27 November: 56 273.87 + 100 = 56 373.87  
1 December: 56 373.87 - 337.6580 = 56 036.21  
4 December: 56 036.21 + 100 = 56 136.21  
5 December: 56 136.21 + 2.93$ = 56 139.14   # Dividends  
11 December: 56 139.14 + 100 = 56 239.14  
13 December: 56 239.14 + 38 958.28 = 95 197.42  
14 December: 95197.62 + 26 371.935 = 121 569.56  
18 December: 121 569.56 + 100 = 121 669.56  
26 December: 121 669.56 + 100 = 121 769.56  
2 January: 121 769.56 - 253.2434 - 426.3514 + 100 = 121 189.76  
5 January: 121 189.76 - 18 185.80 - 36 356.50 = 66 647.46  
8 January: 66 647.46 + 100 = 66 747.46  
15 January: 66 747.46 + 100 = 66 847.46  
22 January: 66 847.46 + 100 = 66 947.46  
24 January: 66 947.46 + 19 507.09 = 86 454.55  
29 January: 86 454.55 + 100 = 86 554.55  
2 February: 86 554.55 - 288.5985 - 422.0472 - 35 936.16 = 49 907.74  
5 February: 49 907.74 + 100 = 50 007.74  
7 February: 50 007.74 + 102 504.435 = 152 512.175  
12 February: 152 512.175 + 100 = 152 612.175  
20 February: 152 612.175 + 100 = 152 712.175  
22 February: 152 712.175 - 56.09 - 20.45 = 152 635.64  
Final amount: 152 635.64$  

# Calculations for Integrative test 3 - ComplexBadStrategy
- Commission: 10$ per trade
- Initial cash: $100 000
- Short interest: 20%
- Minimum sort maintenance margin: 25%
- Liquidation delay: 2 (After two days of margin call, the broker will liquidate positions to cover)

**Calculate after initial trade:**  
**sell NVDA short**: 400 share at 429$  
account cash: 100 000 + 429 * 400 - 10 = 271 590$  
collateral: (435.02 * 400 + 10) * 1.25 = 217 522.5$  
No margin call  

**December 1st: charge interests:**  
interests: 3036.18$  
account cash: 271 590 - 3036.18 = 268 553.82$  
collateral: (467.61 * 400 + 10) * 1.25 = 233 817.5$  
No margin call  

**January 2nd: charge interests:**  
interests: 3381.42$
account cash: 268 553.82 - 3381.42 = 265 172$
collateral: (481.68 * 400 + 10) * 1.25 = 240 852$

**January 9th: Margin Call:**  
account cash: 265 172$  
collateral: (531.40 * 400 + 10) * 1.25 = 265 712.5$  
**Margin call of:** 540.5$

**January 10th: Margin Call day 2:**
account cash: 265 172$  
collateral: (543.5 * 400 + 10) * 1.25 = 271 762.5$
**Margin call of:** 6590.5$

**January 11th: Liquidation orders:**  
Note: The liquidation orders are executed at the opening price of the next day (January 12th)  
account cash: 265 172$  
collateral: (548.22 * 400 + 10) * 1.25 = 274 122.5$
**Margin call of:** 8950.5$
**We are going to withdraw 46 683.40$** in order to make a bankrupt of about 1$

**January 12th: Liquidation orders executed:**
account cash: 265 172.40 - 46 683.40 - (546.2 * 400 + 10) = -1.$  **BANKRUPT**
collateral: 0  
**Margin call of:** 0  

**February 1st: charge interests: -- Won't happen because of the cash we withdrew**  
account cash: -1.00 - 1108.15 = -1109.15$