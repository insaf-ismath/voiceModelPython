import tensorflow as tf
import numpy as np
import urllib

iris_training_set = "iris_trainig.csv"
iris_training_set_url = "http://download.tensorflow.org/data/iris_training.csv"

iris_test_set = "iris_test.csv"
iris_test_set_url = "http://download.tensorflow.org/data/iris_test.csv"


def main():

    #making data set
    raw_data = urllib.urlopen(iris_training_set_url).read()
    with open(iris_training_set,"w") as file:
        file.write(raw_data)

    raw_data = urllib.urlopen(iris_test_set).read()
    with open(iris_test_set,"w") as file:
        file.write(raw_data)


    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=iris_training_set,
      target_dtype=np.int,
      features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = iris_test_set,
        target_dtype = np.int,
        feature_dtype = np.float32
    )

    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]





