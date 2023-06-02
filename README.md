# Detecting Parkinsons with ML
[![Open Word-Level In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pshah123/parkinsons-AI/blob/master/Train.ipynb)

We'll use the data from UC Irvine's amazing dataset repository, specifically the [Parkinsons ML database](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/).

There are two datasets within this. The first is in the root folder (`parkinsons.data` which is included here too) and can be used to detect Parkinsons. The second is within the `telemonitoring/` directory and contains UDPR scores for us to predict.

## Approach

Parkinsons detection islikely best done with an XGBoost since outputs are 0 or 1 and it seems mostly linear.

The UDPR is very hard to fine tune with XGBoost. With an NN in Keras, we can fit much better. There are still some very bad apples in our data/predictions bur the performance is overall/on average much better.

Both approaches are provided in the Jupyter notebook for this repo (`Train.ipynb`). Run using `jupyter notebook` from this repo's root folder in the terminal.

You can find more info on the datasets in UCI's database. ([info for Parkinsons detection](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.names)) ([info for UDPR](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.names))

## Requirements

- Python 3 (I **highly** recommend using Anaconda as this will save you a TON of time)
- XGBoost (`pip install xgboost`)
- Keras (`pip install keras`)
- sklearn (`pip install scikit-learn`)
- Jupyter ([instructions](http://jupyter.org/install))
- Pandas (`pip install pandas`)
- NumPy (`pip install numpy`)
- Tensorflow or another Keras backend (`pip install tensorflow` or for GPU assuming CUDA and CUDNN already installed and in PATH, `pip install tensorflow-gpu`)
