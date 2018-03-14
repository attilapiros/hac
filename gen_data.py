## https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/
## http://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html

from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import numpy

# generate 2d classification dataset
X, labels = make_blobs(n_samples=1000, n_features=2, centers=3)
colors = {0:'red', 1:'blue', 2:'green'}
colored_labels = numpy.vectorize(lambda l: colors[l])(labels)

x_coordinates = X[:, 0] 
y_coordinates = X[:, 1]
pyplot.scatter(x_coordinates, y_coordinates, marker='o', c=colored_labels, s=25, edgecolor='k', label=y)
df = DataFrame(dict(x=x_coordinates, y=y_coordinates, label=labels))
df.to_csv("data.csv")

!hdfs dfs -put -f data.csv
