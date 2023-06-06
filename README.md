DL-COMVIS
==============================

Deep Learning in Computer Vision (June 2023) @ DTU

## Setup

Start by cloning the repository:
```
git clone https://github.com/albertkjoller/DL-COMVIS.git
```
Now move to this repository:
```
cd DL-COMVIS
```

### HPC

Load a Python version and create a virtual environment:
```
module load python3/3.10.7

python3 -m venv venv
```

Now, activate the environment:
```
source venv/bin/activate
```

Now, install the required dependencies:
```
pip3 install -r requirements.txt
```

We need access to Pytorch with GPU support. On the HPC cluster we first activate CUDA:

```
module load cuda/11.8
```

Now install Pytorch with CUDA 11.8 support by;
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Project 1, hotdog

This project is a faithful recreation of [Jian Yangs hotdog app from Silicon Valley](https://www.youtube.com/watch?v=tWwCK95X6go)



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
