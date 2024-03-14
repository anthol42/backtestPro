# BacktestPro
A robust and customizable financial backtesting platform.

## Features
Features are not implemented yet, but will be one day.
- Backtest single or with multiple stocks simultaneously
- Feed a moving window to the strategy
- Multiple time resolution simultaneously
- Take into account:
  - Trading fees
  - Short splits
  - Dividends
- See trades done with Strategy using two engine MatPlotLib or FinCharts.
- Margin trading and short selling
- Customizable

## TODO
- [X] Add a way to dynamically preprocess data to extend the number of time resolutions
- [X] Add a method in the strategy that can preprocess the data to add indicators on the fly.
- [X] Test multiple cases (integrative testing) to see if money and performances are accurately computed
- [X] Test margin investing, margin call and short selling.


## RoadMap
- [X] Backtest Engine
  - [X] Modules (Implementation)
  - [X] Unit testing
  - [X] Integration Testing
  - [X] Documentation
- [X] DataPipeline
  - [X] Modules (Implementation)
  - [X] Unit testing
  - [X] Documentation
- [ ] Indicators
  - [ ] Modules (Implementation)
  - [ ] PineScript to Python translator
  - [ ] Unit testing
  - [ ] Documentation
- [ ] Server Module
  - [ ] Modules (Implementation)
  - [ ] Unit testing
  - [ ] Documentation
- [ ] Integration Testing
- [ ] End-to-end testing
- [ ] Final documentation + Examples
- [ ] Render documentation
- [ ] Alpha Release (In a pip module)
---
Iterative:
- [ ] Bug Fixes
- [ ] Add user requested features
--- 
- [ ] Beta Release
---
Iterative:
- [ ] Bug Fixes
---
- [ ] Stable Release
## Future imrpovements:
- Implement the security measures for a safe baktesting from seeking alpha:
  - URL: https://seekingalpha.com/performance/quant
  - Check Accordion: What are common issues with back testing and performance results?

## [TODO](src/backtest/README.md)



## Useful ressources for the future
Compile C++ wheels for multiple platform using GitHub Actions:  https://cibuildwheel.pypa.io/en/stable/
Package binaries tutorial: https://pybind11.readthedocs.io/en/stable/compiling.html#generating-binding-code-automatically
Automatically compile c++ code (on the fly, but not for packaging): [cppimport](https://github.com/tbenthompson/cppimport)