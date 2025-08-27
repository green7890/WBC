# WBC

Classifying breast cancer cells as malignant or benign based on Wisconsin Breast Cancer Dataset

## Usage

Tested with python 3.13

(If you're new to python, I recommend first running `python3.13 -m venv ~/.virtualenvs/WBCVenv` and then for me (on a mac sequoia 15.5) going into my .zshrc and adding the line `source ~/.virtualenvs/WBCVenv/bin/activate` and then starting with a fresh terminal. There should be only one source...activate line in the zshrc. Make sure ~/.virtualenvs exists and ~/.virtualenvs/WBCVenv doesn't exist, if it does, you could append something to WBCVenv)

Then install packages with `pip install -r /path/to/requirements.txt`

The main usage can be found by running the jupyter notebook `WBC_Classification.ipynb`

Other files that are worth running if one is interested (mainly used during development of the notebook, but these contain finished chunks of code):

- `libraryTest.py`
- `section3.py`
- `section4.py`
- `section5.py`
- `section6.py`
- `section7-2.py`
- `WBCWithPytorchNN3.py` (which is actually tensorflow)

## Credits

Project is based on suggestion number 9 from: https://www.datacamp.com/blog/machine-learning-projects-for-all-levels

WBC (Wisconsin Breast Cancer) dataset found at: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Sections 1-6 are based on a tutorial from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
(adapting it to the WBC dataset)

Section 7 is about Z-Score normalization, from https://www.geeksforgeeks.org/data-normalization-with-pandas/

Section 8 is about using tensorflow implementation of a basic neural network as the ML model, constructed with the help of the tutorial: https://www.tensorflow.org/tutorials/quickstart/beginner