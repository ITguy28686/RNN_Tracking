

import os
from dataset.dataset import Dataset
from urllib.request import urlretrieve
from zipfile import ZipFile
from network.eval import Learning
import tensorflow as tf

import config

FLAGS = tf.app.flags.FLAGS
data_dir = config.data_dir

def main(argv=None):
    Learning()


if __name__ == '__main__':
    tf.app.run()
