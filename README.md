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
- [X] Record the number of active pipes ('get' got called) and give different ids accordingly
- [X] Datapipe automatically detect if the pipe changed (Reset cache)
- [X] Clean serve module
- [X] serve module documentation
- [ ] Make readme more user friendly
- [ ] Licence 

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
- [X] Indicators
  - [X] Modules (Implementation)
  - [X] Unit testing
  - [X] Documentation
- [X] Server Module
  - [X] Modules (Implementation)
  - [X] Prebuilt renderers
  - [X] Unit testing
  - [X] Documentation
- [ ] Final documentation
- [ ] Render documentation
- [ ] Make notebooks for examples (tutorial)
- [ ] Render the notebooks in the documentation
- [ ] Make a github pages for the documentation
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



## Useful ressources for the future
Convert html to pdf: [WeasyPrint](https://weasyprint.org)  
Compile C++ wheels for multiple platform using GitHub Actions:  https://cibuildwheel.pypa.io/en/stable/
Package binaries tutorial: https://pybind11.readthedocs.io/en/stable/compiling.html#generating-binding-code-automatically
Automatically compile c++ code (on the fly, but not for packaging): [cppimport](https://github.com/tbenthompson/cppimport)