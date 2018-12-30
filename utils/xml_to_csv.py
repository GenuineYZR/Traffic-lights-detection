"""
Convert xml to csv.

Example usage:
  python xml_to_csv.py --xml_input=data/input --output_filename=MY_DATA --output_path=data/output

"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('xml_input', None, 'Path to the XML input')
flags.DEFINE_string('output_filename', None, 'Filename of CSV')
flags.DEFINE_string('output_path', os.getcwd(), 'Path to output CSV')
FLAGS = flags.FLAGS

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(_):
    image_path = FLAGS.xml_input
    xml_df = xml_to_csv(image_path)

    output_path = os.path.join(FLAGS.output_path, FLAGS.output_filename + '.csv')
    xml_df.to_csv(output_path, index=None)
    print('Successfully converted xml to csv.')

if __name__ == '__main__':
    tf.app.run()
