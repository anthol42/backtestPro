# FinBackTest
New way to backtest stock trading strategies.

## TODO
- [X] Make Portfolio object
- [ ] Finish Broker object
  - [X] Make that the account does not take a transaction as parameter, but builds the transaction inside method
  - [X] Instead of filling bankruptcy, make a new margin call over no specific security to warns.  If demand not met, liquidate stocks
  - [X] Verify if there is no margin call anymore (If they have been paid)
  - [X] Liquidate short margin calls
  - [ ] Trades
  - [X] Change order in Tick method: trades should be before margins and all that stuff
  - [ ] Handle margin call that expire that are not for tickers (Not enough fund, short margin call, etc)
  - [ ] Handle interest rates week-end calculation
  - [ ] Do not forget to add trading fees (Relative or absolute)
  - [ ] Record Statistics
- [ ] BackTest object
- [ ] Make a way to output stats at the end of simulation
- [ ] Make a way to show trades using matplotlib AND plotly (user can choose engine)