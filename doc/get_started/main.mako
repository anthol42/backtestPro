<div class="row">
    <!-- Table of Contents Section -->
    <div id="tableOfContents" class="col-md-3" style="width: 300px">
        <nav id="toc">
            <ul class="nav flex-column">
                <li class="nav-item">
                    <a class="nav-link" href="#install">Install</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#build-from-source">Build from source</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#Example">Example</a>
                </li>
            </ul>
        </nav>
    </div>
    <div class="toc-doc">
        <div class="col-md-9">
            <div id="install" class="section">
                <h2>Install</h2>
                <p class="code-header">The easiest way to install it is with pip:</p>
                <div class="shell-box">
                    <pre><code>pip install backtestPro</code></pre>
                    <div class="copy-button-container">
                        <button class="copy-button" onclick="copyToClipboard(this)"><i class='fas'>&#xf0c5;</i></button>
                    </div>
                </div>
                <p class="code-header">There are a lot of dependencies that are optionals, but useful to have:</p>
                <ul class="list-group list-group-flush" style="margin-bottom: 15px;">
                    <li class="list-group-item list-group-item-action"><strong>TA-Lib</strong>: A technical analysis
                        library that is used to calculate technical indicators.
                    </li>
                    <li class="list-group-item list-group-item-action"><strong>Plotly</strong>: A plotting library that
                        is used to render charts during the production of reports.
                    </li>
                    <li class="list-group-item list-group-item-action"><strong>WeasyPrint</strong>: A library that
                        converts html to pdf. It is used to render the reports in pdf format.
                    </li>
                    <li class="list-group-item list-group-item-action"><strong>kaleido</strong>: An optional library of
                        Plotly that is used to render the charts in the reports.
                    </li>
                    <li class="list-group-item list-group-item-action"><strong>schedule</strong>: A library that is used
                        to schedule the run of the strategy in production.
                    </li>
                    <li class="list-group-item list-group-item-action"><strong>python-crontab</strong>: A library that
                        is used to schedule the run of the strategy in production.
                    </li>
                </ul>
                <p class="code-header">To install all the optional dependencies with backtest-pro, you can run the
                    following command:</p>
                <div class="shell-box">
                    <pre><code>pip install backtest-pro[optional]</code></pre>
                    <div class="copy-button-container">
                        <button class="copy-button" onclick="copyToClipboard(this)"><i class='fas'>&#xf0c5;</i></button>
                    </div>
                </div>
            </div>
            <div id="build-from-source" class="section">
                <h2>Build from source</h2>
                <p>To get access to the latest features of the framework, you can build it from source. Note that it is
                    not recommended to do this for a production project as the api might change without warning and some
                    features might be unstable.</p>
                <p class="code-header">To install <b>backtest-pro</b> from source, you can clone the repository and
                    install it using pip:</p>
                <div class="shell-box">
                    <pre><code>git clone https://github.com/anthol42/backtestPro.git</code></pre>
                    <div class="copy-button-container">
                        <button class="copy-button" onclick="copyToClipboard(this)"><i class='fas'>&#xf0c5;</i></button>
                    </div>
                </div>
                <p class="code-header">Move to the cloned repository:</p>
                <div class="shell-box">
                    <pre><code>cd backtestPro</code></pre>
                    <div class="copy-button-container">
                        <button class="copy-button" onclick="copyToClipboard(this)"><i class='fas'>&#xf0c5;</i></button>
                    </div>
                </div>
                <p class="code-header">Then, install the package:</p>
                <div class="shell-box">
                    <pre><code>pip install .</code></pre>
                    <div class="copy-button-container">
                        <button class="copy-button" onclick="copyToClipboard(this)"><i class='fas'>&#xf0c5;</i></button>
                    </div>
                </div>
            </div>
            <div id="Example" class="section">
                <h2>Example</h2>
                <p class="code-header">This example shows how to backtest a simple strategy based on technical analysis.
                The strategy buys long when the MACD crosses over the signal line and sells when the MACD crosses under it.
                It is also required that the MACD is below 0 to trigger a buy.  All default backtest parameters are used.
                The backtest is made on the Magnificent 7 stocks from 2010 to 2020.</p>
                <div class="code-container">
                    ${code_example}
                     <div class="copy-button-container">
                        <button class="copy-button" onclick="copyToClipboardText(this, 'code_example')"><i class='fas'>&#xf0c5;</i></button>
                    </div>
                </div>
                <p>For more examples, please check our <a href="tutorials">Tutorials section.</a></p>
            </div>
        </div>
    </div>
     <script>
        const codeTexts = {
            "code_example": '${code_example_raw}'
        };
     </script>
</div>