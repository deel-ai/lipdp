# Purpose of this library :

Conventionally, Differentially Private ML training relies on Gradient Clipping to guarantee verifiable privacy guarantees.
By using 1-Lipschitz networks developped by the deel-lip project. We can propose a new alternative to gradient clipping based
DP ML. Indeed, by theoretically bounding the value of the sensitivity of our 1-Lipschitz layers, we can directly calibrate a
batchwise noising of the gradients to guarantee (epsilon,delta)-DP.

Therefore, the computation time is heavily reduced and the results on the MNIST and CIFAR10 datasets are the following :


# Status of the repository : 

- ci tests to develop.
- sensitivity.py to debug.
- requirements.txt tested on my machine, still to check by someone else.

# Deel library repository template

Ce dépôt git sert de template pour les librairies DEEL ayant vocation à être rendues publiques sur github.
Il donne la structure des répertoires d'un projet telle que celle adoptée par les librairies DEEL déjà publiques.

A la racine du projet on trouve:

- deel : répertoire destiné à recevoir le code de la librairie. C'est le premier mot de l'espaces de nommage de
        la librairie. Ce n'est pas un module python, il ne contient donc pas de fichier __init__.py.
        Il contient le module principal de la librairie du nom de cette librairie.
        
        Example: 
        
        librairie **deel-lip**:
                    deel/deel-lip       

- docs: répertoire destiné à la documentation de la librairie

- tests: répertoire des tests unitaires

- .pre-commit-config.yaml : configuration de outil de contrôle avant commit (pre-commit)

- LICENCE/headers/MIT-Clause.txt : entête licence MIT injectée dans les fichiers du projet

- CONTRIBUTING.md: description de la procédure pour apporter une contribution à la librairie.

- GOUVERNANCE.md: description de la manière dont la librairie est gérée.

- LICENCE : texte de la licence sous laquelle est publiée la librairie (MIT).

- README.md 


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

# README sections

Conventionally, Differentially Private ML training relies on Gradient Clipping to guarantee verifiable privacy guarantees.
By using 1-Lipschitz networks developped by the deel-lip project. We can propose a new alternative to gradient clipping based
DP ML. Indeed, by theoretically bounding the value of the sensitivity of our 1-Lipschitz layers, we can directly calibrate a
batchwise noising of the gradients to guarantee (epsilon,delta)-DP.


## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [👨‍🎓 Creator](#-creator)
- [🗞️ Citation](#-citation)
- [📝 License](#-license)

## 🔥 Tutorials



## 🚀 Quick Start

Libname requires some stuff and several libraries including Numpy. Installation can be done using Pypi:

```python
pip install dist/lipdp-0.0.1a0-py2.py3-none-any.whl[dev]
```

Now that lipdp is installed, here are some basic examples of what you can do with the
 available modules.

## 📦 What's Included

Code can be found in the `lipdp` folder, the documentation ca be found by running
 `mkdocs build` and `mkdocs serve` (or loading `site/index.html`). Experiments were
  done using the code in the `experiments` folder.

## 👍 Contributing

Feel free to propose your ideas or come and contribute with us on the Libname toolbox! We have a specific document where we describe in a simple way how to make your first pull request: [just here](CONTRIBUTING.md).

### pre-commit : Conventional Commits 1.0.0

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


## 🙏 Acknowledgments


## 👨‍🎓 Creators

If you want to highlights the main contributors


## 🗞️ Citation



## 📝 License

The package is released under [MIT license](LICENSE).
