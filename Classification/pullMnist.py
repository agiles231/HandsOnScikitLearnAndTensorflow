
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt

def pull_mnist():
    return fetch_mldata('MNIST original')