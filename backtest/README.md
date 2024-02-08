# FinBackTest
New way to backtest stock trading strategies.

## TODO
- [X] Make Portfolio object
- [X] Finish Broker object
  - [X] Make that the account does not take a transaction as parameter, but builds the transaction inside method
  - [X] Instead of filling bankruptcy, make a new margin call over no specific security to warns.  If demand not met, liquidate stocks
  - [X] Verify if there is no margin call anymore (If they have been paid)
  - [X] Liquidate short margin calls
  - [X] Evaluate collateral value (method) [read this](https://www.investopedia.com/ask/answers/05/shortmarginrequirements.asp)
  - [X] Implement delete_margin_call method and use it in '_get_short_collateral' and '_get_long_collateral'
  - [X] Trades (Do not forget to add initial value of position as debt in short selling)
  - [X] Do not forget to add trading fees (Relative or absolute)
  - [X] Change order in Tick method: trades should be before margins and all that stuff
  - [X] Handle margin call that expire that are not for tickers (Not enough fund, short margin call, etc)
  - [X] Handle interest rates week-end calculation
  - [X] Handle bankruptcy if all positions are sold and there is not enough funds to pay debt.
  - [X] Record Statistics
    - [X] Total value of portfolio
    - [X] Trades stats (duration and profits)
  - [X] Handle dividends
- [ ] BackTest object
  - [X] Handle stock splits in init section of run method
  - [X] Make the step function
    - [X] Find a way to prepare data of higher resolution than main timestep resolution.
    - [X] If For the current timestep, the main resolution series is all nan, ignore stock.
  - [X] Implement backtestResult class to save all stats and run info (Plus saving method)
    - [X] Make exports methods in child class
    - [X] Make metadata class
    - [X] Enable loading the whole state from a backtest file
  - [X] Implement saving and stats recording in backtest class (run method)
- [X] Make a way to show trades using matplotlib
- [ ] Unit tests.  Make sure to test all edge cases.
  - [ ] Make the portfolio handle its debt and returns (Handle short + implement in broker)
- [X] Package all stats, run info, meta data, config, states over the simulation and debug info in a json file format
- [ ] FinCharts flutter GUI (Other charting engine)