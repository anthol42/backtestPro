# How to contribute
*Note: This document has been inspired by the 
[Contributing to Pandas](https://pandas.pydata.org/docs/development/contributing.html#tips-for-a-successful-pull-request) guide.*

## Table of contents
- [Bug report and feature request](#Bug-report-and-feature-request)
- [Finding an issue to contribute to](#Finding-an-issue-to-contribute-to)
- [Submitting a pull request](#Submitting-a-pull-request)
  - [Version control, Git, and GitHub](#Version-control-Git-and-GitHub)
  - [Fork the repository](#Fork-the-repository)
  - [Creating a feature branch](#Creating-a-feature-branch)
  - [Making changes](#Making-changes)
  - [Pushing your changes](#Pushing-your-changes)
  - [Making a pull request](#Making-a-pull-request)
  - [Updating your pull request](#Updating-your-pull-request)
  - [Updating the development environment](#Updating-the-development-environment)
- [Contribution Guidelines](#Contribution-Guidelines)
  - [Code standards](#Code-standards)
  - [Optional dependencies](#Optional-dependencies)
  - [Documentation](#Documentation)
  - [Type hints](#Type-hints)
  - [Testing](#Testing)
- [Commit Convention](#Commit-Convention)

## Bug report and feature request
Bug reports and feature requests are an important part of the development process.  If you find a bug or have a feature 
request, please open an issue on the [issue tracker](https://github.com/anthol42/backtestPro/issues).  For bug requests,
please provide an example to reproduce the bug.  Please provide as much information as possible to help us understand the
issue.  Also provide you system information (OS, Python version, package version, etc.)  For feature requests, please 
provide a detailed description of the feature you would like to see.  Also, provide a code example of the desired
behavior of the requested feature.

## Finding an issue to contribute to
If you are new to the project or to open source in general, you can start by looking at the issues labeled as
[good first issue](https://github.com/anthol42/backtestPro/labels/good%20first%20issue) or 
[help wanted](https://github.com/anthol42/backtestPro/labels/help%20wanted).  These issues are usually easier to solve 
and are a good way to get started with the project.

## Submitting a pull request
If you want to contribute to the project, you can submit a pull request.  To do so, you can follow these steps:

### Version control, Git, and GitHub
The backtest-pro project uses Git for version control and GitHub for hosting the repository.  To contribute, you will 
need to sign in a free GitHub account. If you are not familiar with Git or GitHub, you can learn more about it by 
reading the [GitHub documentation](https://docs.github.com/en) or the [Git documentation](https://git-scm.com/doc), or 
by following a tutorial on the web.

Also, the project uses a forking workflow.  To contribute, you will need to fork the repository, make your changes, and
submit a pull request.  If you are not familiar with the forking workflow, you can read the following of this guide.

Below are some useful resources for learning more about forking and pull requests on GitHub:

the [GitHub documentation for forking a repo.](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
the [GitHub documentation for collaborating with pull requests.](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)
the [GitHub documentation for working with forks.](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)

### Fork the repository
You will need your own copy of the backtest-pro project to make changes.  To do so, you can fork the repository by
clicking on the "Fork" button on the top right of the [main repository page](https://github.com/anthol42/backtestPro).  
Please uncheck the box to copy only the main branch before selecting Create Fork. You will want to clone your fork to 
your machine.  You can clone the fork on your machine using the following command:

```shell
git clone https://github.com/your-user-name/backtestPro backtestPro-yourname
cd backtestPro-yourname
git remote add upstream https://github.com/anthol42/backtestPro
git fetch upstream
```

This creates the directory backtestPro-yourname and connects your repository to the upstream (main project) 
backtestPro repository.

### Creating a feature branch
Your local main branch should always reflect the current state of backtestPro repository. First ensure it’s up-to-date 
with the main backtestPro repository.

```shell
git checkout main
git pull upstream main --ff-only
```

Then, create a feature branch for making your changes. For example
```shell
git checkout -b shiny-new-feature
```
This changes your working branch from main to the shiny-new-feature branch. Keep any changes in this branch specific to 
one bug or feature, so it is clear what the branch brings to backtestPro. You can have many feature branches and switch 
in between them using the git checkout command.

When you want to update the feature branch with changes in main after you created the branch, check the section on 
[updating a PR](#Updating-your-pull-request).

### Making changes
Before modifying any code, ensure you set up an appropriate development environment. To do so, create a virtual 
environment with python 3.8 or higher and install the required dependencies.  You can install the dependencies by
running the following command:

```shell
pip install requirements.txt
```

Then, once you have made code changes, you can see all the changes you’ve currently made by running.

```shell
git status
```

For files you intended to modify or add, run:
```shell
git add path/to/file-to-be-added-or-changed.py
```
or, to add all changes:
```shell
git add .
```
Running ```git status``` again should display the changes you’ve made:
```text
On branch shiny-new-feature

     modified:   /relative/path/to/file-to-be-added-or-changed.py
```
Finally, commit your changes to your local repository with an explanatory commit message

```shell
git commit -m "Feat(shiny-new-feature): description of the feature"
```

To know how to write a good commit message, you can read the [Commit Convention](#Commit-Convention) section.


### Pushing your changes
Once you have committed your changes, you can push them to your fork on GitHub.  You can do so by running the following
command: 
```shell
git push origin shiny-new-feature
```
Here ```origin``` is the default name given to your remote repository on GitHub. You can see the remote repositories
```
git remote -v
```
If you added the upstream repository as described above you will see something like
```text
origin  git@github.com:yourname/backtestPro.git (fetch)
origin  git@github.com:yourname/backtestPro.git (push)
upstream        git://github.com/anthol42/backtestPro.git (fetch)
upstream        git://github.com/anthol42/backtestPro.git (push)
```
Now your code is on GitHub, but it is not yet a part of the backtestPro project. For that to happen, a pull request 
needs to be submitted on GitHub.

### Making a pull request
Once you have finished your code changes, your code change will need to follow the 
[backtestPro contribution guidelines](#Contribution-Guidelines) to be successfully accepted.

If everything looks good, you are ready to make a pull request. A pull request is how code from your local repository 
becomes available to the GitHub community to review and merged into project to appear the in the next release. To submit 
a pull request:
1. Navigate to your repository on GitHub.
2. Click on the ```Compare and pull request``` button.
3. You can then click on ```Commits``` and ```Files Changed``` to make sure everything looks okay one last time
4. Write a descriptive title that includes prefixes. backtestPro uses a convention for title prefixes. Here are some 
common ones along with general guidelines for when to use them:
   - ENH: Enhancement, new functionality
   - BUG: Bug fix
   - DOC: Additions/updates to documentation
   - TST: Additions/updates to tests
   - PERF: Performance improvement
   - TYP: Type annotations
   - CLN: Code cleanup
5. Write am exhaustive description of your changes in the Preview Discussion tab.
6. Click the ```Send Pull Request``` button.

This request then goes to the repository maintainers, and they will review the code.

### Updating your pull request
Based on the review you get on your pull request, you will probably need to make some changes to the code. You can 
follow the code committing steps again to address any feedback and update your pull request.

It is also important that updates in the backtestPro main branch are reflected in your pull request. To update your 
feature branch with changes in the backtestPro main branch, run:
```shell
git checkout shiny-new-feature
git fetch upstream
git merge upstream/main
```
If there are no conflicts (or they could be fixed automatically), a file with a default commit message will open, and 
you can simply save and quit this file.

If there are merge conflicts, you need to solve those conflicts. See for example at
https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/ for an explanation on how to do 
this.

Once the conflicts are resolved, run:

1. ```git add -u``` to stage any files you’ve updated
2. ```git commit``` to finish the merge.

After the feature branch has been update locally, you can now update your pull request by pushing to the branch on 
GitHub:
```shell
git push origin shiny-new-feature
```

### Updating the development environment
It is important to periodically update your local main branch with updates from the backtestPro main branch and update 
your development environment to reflect any changes to the various packages that are used during development.
```shell
git checkout main
git fetch upstream
git merge upstream/main
# activate the virtual environment based on your platform
python -m pip install --upgrade -r requirements-dev.txt
```


## Contribution Guidelines

### Code standards
Writing good code is not only about what code you write, but also how you write it.  This is why we built a set of rules
to write code that is easy to read and maintain.  Please follow these rules when contributing to the project.

### Optional dependencies
The backtest-pro project has a few optional dependencies that are not installed by default.  If you want to add another
optional dependency, please add it to the optional dependencies in the setup.py file.  Also, wrap the import of the
module in a try except block to avoid import errors when the module is not installed.

### Documentation
Documentation is an important part of the project.  If you are adding a new feature or fixing a bug, please update the
documentation to reflect the changes.  The documentation is written in the reStructuredText format.  The documentation
is the docstrings of the functions and classes.  It is later rendered using Sphinx.  If you are adding a new feature,
please add a new section in the documentation to explain how to use the feature.  If you are fixing a bug, please update
the documentation to reflect the changes.  The documentation will be rendered when a new version is released.

### Type hints
The backtest-pro project uses type hints to make the code more readable and maintainable in functions/methods parameters
and return types.  Please add type hints to the functions and methods you are adding or modifying.

### Testing
The backtest-pro project uses unittests to test the code.  If you are adding a new feature or fixing a bug, please add
 or update the appropriate tests.  The tests are located in the test directory.  Before creating a pull request, make
sure that all the tests pass.  You can run the tests using the following command:

```shell
python -m unittest discover test/<modulename>_tests
```
Example:
```shell
python -m unittest discover test/data_tests
```

## Commit Convention
The know what a given commit is about, we use a commit convention.  A commit message should follow this structure:  
```<type>(<name>): <description>```

where:
- ```<type>``` is the type of the commit.  It can be one of the following:
  - ```Feat``` Adding a new feature or functionality.
  - ```Imp``` Improving a feature or functionality.
  - ```Fix``` Fixing a bug.
- ```<name>``` is the name of the feature or functionality that is being added, improved or fixed.
- ```<description>``` A short description explaining what the commit is about.

Example:  
```Feat(LoadCSV): Add a new datapipe to load data from a CSV file.```

## Tips for a successful pull request
If you have made it to the [Making a pull request](#making-a-pull-request) phase, one of the core contributors may take 
a look. Please note however that a handful of people are responsible for reviewing all the contributions, which can 
often lead to bottlenecks.

To improve the chances of your pull request being reviewed, you should:
- **Follow the [Contribution Guidelines](#Contribution-Guidelines)**.
- **Reference the issue** you are addressing in the pull request.
- **Keep your pull requests small.**  If you are adding a new feature, consider breaking it down into smaller parts and 
  submitting each part as a separate pull request.
- **Write a good commit message.**  A good commit message should explain what the commit is about.  It should be clear and 
  concise.  It should also follow the [Commit Convention](#Commit-Convention).