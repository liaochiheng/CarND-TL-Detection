r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python bosch/create_bosch_tf_record.py \
        --yaml_path=/home/user/bosch/dataset_rgb/train.yaml \
        --output_path=/home/user/bosch.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import sys
import yaml

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
# flags.DEFINE_string('data_dir', '../data/Bosch_Small_TrafficLight_Dataset/dataset_rgb', \
# 					'Directory to dataset.')
flags.DEFINE_string('yaml_path', '../data/Bosch_Small_TrafficLight_Dataset/dataset_rgb/train.yaml', 'Filename of train.yaml.')
flags.DEFINE_string('label_map_path', '../data/Bosch_Small_TrafficLight_Dataset/dataset_rgb/bosch_label_map.pbtxt', \
                    'Path to label map proto')
flags.DEFINE_string('output_path', '../data/Bosch_Small_TrafficLight_Dataset/rgb.record', \
					'Path to output TFRecord')
FLAGS = flags.FLAGS

def dict_to_tf_example(data, label_map_dict):
    """Convert XML derived dict to tf.Example proto.  

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.  

    Returns:
      example: The converted tf.Example.  

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    with tf.gfile.GFile( data['path'], 'rb' ) as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO( encoded_png )
    image = PIL.Image.open( encoded_png_io )
    key = hashlib.sha256( encoded_png ).hexdigest()  

    width = 1280
    height = 720

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for box in data['boxes']:
        xmin.append( float( box['x_min']) / width )
        ymin.append( float( box['y_min']) / height )
        xmax.append( float( box['x_max']) / width )
        ymax.append( float( box['y_max']) / height )
        classes_text.append( box['label'].encode('utf8') )
        classes.append( label_map_dict[ box['label'] ] )

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['path'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['path'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature( encoded_png ),
        'image/format': dataset_util.bytes_feature( 'png'.encode('utf8') ),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def main(_):
    yaml_path = FLAGS.yaml_path

    # output_dir = os.path.dirname( FLAGS.output_path )
    # if not os.path.exists( output_dir ):
    #     os.mkdir( output_dir )
    writer = tf.python_io.TFRecordWriter( FLAGS.output_path )  

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)  

    logging.info('Reading from bosch dataset.')
    print('Reading from bosch dataset.') 

    # images = yaml.load(open(yaml_path, 'rb').read())
    with tf.gfile.GFile( yaml_path, 'r' ) as fid:
        yaml_str = fid.read()
    images = yaml.load( yaml_str )

    for idx, image in enumerate( images ):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len( images ))
            print( 'On image {} of {}'.format( idx, len( images ) ) )

        image['path'] = os.path.join(os.path.dirname( yaml_path ), image['path'])
        
        tf_example = dict_to_tf_example( image, label_map_dict )
        
        writer.write(tf_example.SerializeToString())    
        
    writer.close()


if __name__ == '__main__':
    tf.app.run()