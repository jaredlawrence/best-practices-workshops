# best-practices-workshops

## Workshop Materials
This repository was created for a Best Practices Workshop series hosted by Pitt's student chapter of INFORMS. The content was as follows with the slides linked
* [Version Control with Git and GitHub](https://docs.google.com/presentation/d/e/2PACX-1vT6DykUfm316uvrN8jV4z6k0iM-3QxfVvx2VzPeOSnE8m35mwzAowkAsvp88963VAR2x6-InkB3zrt8/pub?start=false&loop=false&delayms=3000) - Track changes, collaborate safely, and recover from mistakes.
* [Object-Oriented Programming](https://docs.google.com/presentation/d/e/2PACX-1vQuA625kFGva24u2kX7DFqP6FujiEKhYT-TtI8RvZ04aEKrScnD_JhGqonJ3EC96Q/pub?start=false&loop=false&delayms=3000) - Structure reusable code with classes, inheritance, and clean abstract interfaces.
* [Writing Reliable Code](https://docs.google.com/presentation/d/e/2PACX-1vRWBGJqQAtODsJ8_bzip97Lc-vYGAlVzcsxtccz4ZfoyMQs45MsYP73nVp4rWYnWvH56kAoYeddIFG4/pub?start=false&loop=false&delayms=3000) - Make code testable and readable with unit tests, type hints, docstrings, and command-line arguments.
* [Modular Project Design](https://docs.google.com/presentation/d/e/2PACX-1vQg4HOUGQSO7AARI_2X86n1uuTzLWa3kpbNeE2oTMitPUZXyRUzCLmtI8ivNC63MpEBhEW8Fz8IqWew/pub?start=false&loop=false&delayms=3000) - Organizing a full project (i.e., this repository).

The first two presentations were prepared by Prem Shenoy while the latter two were prepared by Jared Lawrence and focused on creating this repository. 

Source material for the first workshop included [First Contributions](https://github.com/firstcontributions/first-contributions) and a previous workshop by Luca Wrabetz. Harvard's [CS50 Python](https://cs50.harvard.edu/python/) was used as source material for the second and third workshops with modifications. The last workshop focusing on this repository was original content.

## What is this repository?
This repository is meant to be a somewhat minimal example for how a project can be well structured; a boilerplate that can be referenced later. Two optimization methods, gradient descent with and without momentum, are compared on two problem types, quadratic and Rosenbrock. All of the experiments can be replicated by running a single shell script. Raw CSVs for each optimization run are produced and figures comparing them are generated. The way things are structured, it is also easy to add additional optimizers and problems without making many changes.

Disclaimer: The vast majority of the code was created using LLMs. I make no claims that it is correct.

## Commands to get the code and replicate the environment locally
Step 1 - Clone the repository
```bash
git clone https://github.com/<your-username>/best-practices-workshops.git
cd best-practices-workshops
```

Step 2 - Create and activate a virtual environment

```PowerShell
# Windows
python -m venv venv
venv\Scripts\activate
```

```bash
# Mac/Linux
python -m venv venv
source venv/bin/activate
```

Step 3 - Install dependencies
```bash
pip install -e ".[dev]"
```

Step 4 - Verify everything works
```bash
pytest -v
```
If all tests pass, you are ready to go.

## Running the experiments
```bash
bash scripts/run_experiments.sh
```

## Adapting this template
### Add a new optimizer
Create src/my_optimizer.py inheriting from Optimizer.
Add one entry each to OPTIMIZERS and OPTIMIZERS_STEMS in scripts/run_benchmark.py

### Add a new problem
Create problem/my_problem.py inheriting from Problem.
Add one entry each to PROBLEMS and PROBLEM_STEMS in script/run_benchmark.py

