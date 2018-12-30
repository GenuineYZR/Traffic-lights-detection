"""Split .csv file for training and validation.

Example usage:
  python csv_dataset_split.py --csv_input=data/data.csv  --split_ratio=0.6 --output_path=data/output

"""

import os
import pandas as pd
import numpy as np 
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('csv_input', None, 'Path to the CSV input')
flags.DEFINE_float('split_ratio', 0.6, 'Ratio of the training set')
flags.DEFINE_string('output_path', os.getcwd(), 'Path to output split CSV')
FLAGS = flags.FLAGS

def main(_):
    df = pd.read_csv(FLAGS.csv_input) # Create a DataFrame object
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= FLAGS.split_ratio

    train = df[msk]
    val = df[~msk]

    training_output_path = os.path.join(FLAGS.output_path, 'Training_set.csv')
    validation_output_path = os.path.join(FLAGS.output_path, 'Validation_set.csv')

    train.to_csv(training_output_path, index=False)
    val.to_csv(validation_output_path, index=False)

    print('Split done')

if __name__ == '__main__':
    tf.app.run()