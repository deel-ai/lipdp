# Contributing

Thanks for taking the time to contribute!

From opening a bug report to creating a pull request: every contribution is
appreciated and welcome. If you're planning to implement a new feature or change
the api please create an [issue first](https://https://github.com/deel-ai/dp-lipschitz/issues/new). This way we can ensure that your precious
work is not in vain.


## Setup with make

- Clone the repo `git clone https://github.com/deel-ai/dp-lipschitz.git`.
- Go to your freshly downloaded repo `cd lipdp`
- Create a virtual environment and install the necessary dependencies for development:

  `make prepare-dev && source lipdp_dev_env/bin/activate`.

Welcome to the team !


## Tests

To run test `make test`
This command activate your virtual environment and launch the `tox` command.


`tox` on the otherhand will do the following:
- run pytest on the tests folder with python 3.6, python 3.7 and python 3.8
> Note: If you do not have those 3 interpreters the tests would be only performs with your current interpreter
- run pylint on the deel-datasets main files, also with python 3.6, python 3.7 and python 3.8
> Note: It is possible that pylint throw false-positive errors. If the linting test failed please check first pylint output to point out the reasons.

Please, make sure you run all the tests at least once before opening a pull request.

A word toward [Pylint](https://pypi.org/project/pylint/) for those that don't know it:
> Pylint is a Python static code analysis tool which looks for programming errors, helps enforcing a coding standard, sniffs for code smells and offers simple refactoring suggestions.

Basically, it will check that your code follow a certain number of convention. Any Pull Request will go through a Github workflow ensuring that your code respect the Pylint conventions (most of them at least).

## Submitting Changes

After getting some feedback, push to your fork and submit a pull request. We
may suggest some changes or improvements or alternatives, but for small changes
your pull request should be accepted quickly (see [Governance policy](https://github.com/deel-ai/lipdp/blob/master/GOVERNANCE.md)).

Something that will increase the chance that your pull request is accepted:

- Write tests and ensure that the existing ones pass.
- If `make test` is succesful, you have fair chances to pass the CI workflows (linting and test)
- Follow the existing coding style and run `make check_all` to check all files format.
- Write a [good commit message](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) (we follow a lowercase convention).
- For a major fix/feature make sure your PR has an issue and if it doesn't, please create one. This would help discussion with the community, and polishing ideas in case of a new feature.

# Deel library repository template

This git repository is a template for the DEEL libraries to be made public on github. The following section
gives the repository's structure. 

At the root of the repository we can find : 

- deel : repository destined to receive the source code of the library. It is the first work of the naming 
        space of the library. It is not a python module therefore does not contain a __init__.py file.
        It contains the eponymous principal module of the library.

        Examples : 

        library **deel-lip** :
                  deel/deel-lip
- docs : repository destined to receive the library's documentation.

- tests : repository of unit tests. 

- .pre-commit-config.yaml : configuration of the pre-commit tool.

- LICENCE/headers/MIT-Clause.txt : MIT license header injected into the project files. 

- CONTRIBUTING.md : description of the contribution process.

- GOUVERNANCE.md: description of the rules that apply to said library.

- LICENCE : Licence text applying to the library (MIT).

- README.md : Informative content.


# pre-commit : Conventional Commits 1.0.0

The commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

```

The commit contains the following structural elements, to communicate intent to the consumers of your library:

- fix: a commit of the type fix patches a bug in your codebase (this correlates with PATCH in Semantic Versioning).

- feat: a commit of the type feat introduces a new feature to the codebase (this correlates with MINOR in Semantic Versioning).

- BREAKING CHANGE: a commit that has a footer BREAKING CHANGE:, or appends a ! after the type/scope, introduces a breaking API change (correlating with MAJOR in Semantic Versioning). A BREAKING CHANGE can be part of commits of any type.

- types other than fix: and feat: are allowed, for example @commitlint/config-conventional (based on the the Angular convention) recommends *build:, chore:, ci:, docs:, style:, refactor:, perf:, test:*, and [others](https://delicious-insights.com/fr/articles/git-hooks-et-commitlint/).
 
- footers other than BREAKING CHANGE: <description> may be provided and follow a convention similar to git trailer format.

- Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE). A scope may be provided to a commit’s type, to provide additional contextual information and is contained within parenthesis, e.g., feat(parser): add ability to parse arrays.


