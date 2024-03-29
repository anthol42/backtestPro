# Backtest-pro: feature rich backtesting framework

## What it is?
**backtest-pro** is a framework that provides a way to test complex strategies in an environment that is designed to 
look as much as possible to the real world.  This way, the historical results are more likely to reflect the
real world results. It is an **event-driven** backtesting framework that is designed to be as flexible as possible and
as complete as possible.  It supports **end-to-end** quant pipeline from data fetching to production release.  
**Also, it has the broader goal of becoming the most complete backtesting framework available for python finely tuned 
for professional applications.**

## Important notice
**backtest-pro** is still in development and is not ready for production use.  There may be bugs that could
make the results of the backtest invalid.  Always double-check the results with your own code.  If you find a bug, please
open an issue on the github page.  The api might also change without notice.

## Table of Contents
- [Main Features](#features)
- [Installation](#installation)
- [Installation from source](#installation-from-source)
- [License](#license)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Features
Here are just a few of the features that **backtest-pro** offers:
- **DataPiplines**
  - An easy to use data pipeline api that makes building a data pipeline a breeze.
  - A pipeline built with the api is easy to maintain and easy to understand.
  - Support caching for more efficient pipelines.
- **Backtest**
  - Backtest with a single or with multiple assets simultaneously
  - Feed a moving window to the strategy
  - Multiple time resolution simultaneously
  - Take into account:
    - Trading fees
    - Margin rates
    - Margin calls
    - Stock splits
    - Dividends
  - Records a lot of metrics for easier analysis and debugging.
- **Release**
  - Use the same code as used in the backtest in production.
  - Only a few lines of codes are necessary to build a production pipeline that can run on a server or locally.
  - Automatic reporting of the results using report builders.  (Html, pdf)
  - Easy integration with other services such as api for algorithmic trading.

## Installation
To install **backtest-pro**, you can use pip:
```commandline
pip install backtest-pro
```
There a few dependencies that are not installed by default.  They are:
- TA-Lib: A technical analysis library that is used to calculate technical indicators.
- Plotly: A plotting library that is used to render charts in the production of reports.
- WeasyPrint: A library that is used to convert html to pdf.  It is used to render the reports in pdf format.
- kaleido: An optional library of Plotly that is used to render the charts in the reports.
- schedule: A library that is used to schedule the run of the strategy in production.
- python-crontab: A library that is used to schedule the run of the strategy in production.

To install the dependencies, you can use the following command:
```commandline
pip install backtest-pro[optional]
```

## Installation from source
To install **backtest-pro** from source, you can clone the repository and install it using pip:
```commandline
git clone https://github.com/anthol42/backtestPro.git
```
Move to the cloned repository:
```commandline
cd backtestPro
```
Then, you can install with the following command:
```commandline
pip install .
```

## License
[GNU General Public License v3.0](LICENSE)

## Documentation
*Not there yet, comming as soon as possible:)*

## Contributing
All contributions are welcome.  It can be a bug report, a bug fix, a new feature, a documentation improvement, etc.  
If you want to contribute, please read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Do not forget to add tests for your code and to run all the tests before submitting a pull request.

As contributors and maintainers to this project, you are expected to abide by pandas' code of conduct. More information 
can be found at: [Contributor Code of Conduct](CODE_OF_CONDUCT.md)

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