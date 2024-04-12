<nav class="navbar navbar-light navbar-expand bg-light fixed-top shadow-sm" id="navbar-main">
    <div class="navbar-container" style="padding-left: 20px">
            <a class="navbar-brand logo" href="${absolute_path}">
                <img alt="Logo image" class="logo__image only-light" src="${absolute_path}/assets/logo_light.svg">
            </a>
            <div class="col-lg-9 navbar">
                <div class="mr-auto" id="navbar-center">

                    <div class="navbar-center-item">
                        <ul class="navbar-nav" id="navbar-main-elements">
                            <li class="toctree-l1 nav-item">
                                <a class="reference internal nav-link ${'active' if page_id == 'home' else ''}" href="${absolute_path}">
                                    Home
                                </a>
                            </li>

                            <li class="toctree-l1 nav-item">
                                <a class="reference internal nav-link ${'active' if page_id == 'get_started' else ''}" href="${absolute_path}/get_started.html">
                                    Get Started
                                </a>
                            </li>

                            <li class="toctree-l1 nav-item">
                                <a class="reference internal nav-link" href="${absolute_path}/tutorials">
                                    Tutorials
                                </a>
                            </li>

                            <li class="toctree-l1 nav-item">
                                <a class="reference internal nav-link" href="${absolute_path}/docs/backtest">
                                    Documentation
                                </a>
                            </li>


                            <li class="nav-item">
                                <a class="nav-link nav-external ${'active' if page_id == 'about' else ''}" href="${absolute_path}/about.html">About</a>
                            </li>

                        </ul>
                    </div>

                </div>

                <div id="navbar-end">

                    <div class="navbar-end-item">
                        <code class="navbar-version">v1.21.0</code>
                    </div>

                    <div class="navbar-end-item">
                        <ul aria-label="Icon Links" class="navbar-nav" id="navbar-icon-links">
                            <li class="nav-item">
                                <a class="nav-link" href="https://github.com/anthol42/backtestPro" rel="noopener" target="_blank"
                                   title="GitHub"><img alt="GitHub" class="icon-link" src="${absolute_path}/assets/github-mark.svg"
                                                       style="height: 1.2em; margin: 0px; padding: 0px;"></a>
                            </li>
                        </ul>
                    </div>

                </div>
            </div>
    </div>
</nav>