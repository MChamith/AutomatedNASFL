This repository holds code for automated neural architecture search and hyperparameter optimization for federated learning using large language models. Here we utilize open ai chatgpt API for suggesting neural architectures for neural architecture search process. Hyper parameter optimization is done using selective halving method.

### dependencies

1. [pytorch](https://pytorch.org/) 2.0 or latest
2. [torchvision](https://pypi.org/project/torchvision/)
3. [numpy](https://numpy.org/install/) (pip install numpy)
4. [matplotlib](https://matplotlib.org/stable/install/index.html)
5. [openai](https://pypi.org/project/openai/) (pip install openai)
6. [sklearn](https://scikit-learn.org/stable/install.html)
7. [prettytable](https://pypi.org/project/prettytable/)

Once dependencies are installed run *python run_nas.py* for running the NAS/HPO process. Once the a good model and learning rate is found using the search process modify *test.py* with model file location and learning rates found. Then run *test.py* to test the performance of the found model.