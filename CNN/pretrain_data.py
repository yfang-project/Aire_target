"""
Modify the Basenji data for pre-training

@author: Yuan Fang
"""
import os
import sys
import random

import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()

original_path = str(sys.argv[1])
os.chdir(original_path)

# List all the tfrecord files
tf_files = os.listdir('./')
tf_files = [os.path.join('./', file) for file in tf_files if file.endswith('.tfr')]

# Create TF Dataset
def decode_fn(record_bytes):
    features = {
        "sequence": tf.io.FixedLenFeature([], dtype=tf.string),
        "target": tf.io.FixedLenFeature([], dtype=tf.string)
        }
    parsed_example = tf.io.parse_single_example(record_bytes, features)
    
    seq = tf.io.decode_raw(parsed_example['sequence'], tf.uint8)
    targets = tf.io.decode_raw(parsed_example['target'], tf.float16)
    
    return {'sequence': seq, 'target': targets}

def create_byte_feature(values):
    """Convert numpy arrays to bytes features."""
    #values = values.flatten().tostring()
    values = values.flatten().tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
    
    
def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

sample_num = 0
seq_depth = 4 # one-hot encoding 
target_length = 896 
target_num = 1643 
crop_bp = 8192
new_seq_len = 2048
new_target_len = new_seq_len//128 # 16
sampled_position = 5

# Define options
tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

# Modify Basenji data
for dset_file in tf_files:
    dset = tf.data.TFRecordDataset([dset_file], 
                                   compression_type='ZLIB').map(decode_fn)
    new_dset_file = dset_file.strip('trf') + 'tfrecords'

    with tf.io.TFRecordWriter(new_dset_file, tf_opts) as writer:
        for sample in dset:
            seq = sample['sequence']
            target = sample['target']
            seq = seq.numpy().reshape((-1,seq_depth)) 
            target = target.numpy().reshape((target_length,-1)) 
            
            # crop and reshape the input seq
            seq = seq[crop_bp:-crop_bp,:] 
            seq = seq.reshape((-1, new_seq_len, seq_depth)) 
            
            half_len = seq.shape[0]//2 
            kept_ids = random.sample(list(range(half_len//2, seq.shape[0]-half_len//2)), sampled_position)
            
            seq = seq[kept_ids]
            target = target.reshape((-1, new_target_len, target_num)) 
            target = target[kept_ids]
        
            for new_seq,new_target in zip(list(seq),list(target)):
                # hash to bytes
                features_dict = {
                    'sequence': create_byte_feature(new_seq),
                    'target': create_byte_feature(new_target)
                    }
    
                # write example
                example = tf.train.Example(features=tf.train.Features(feature=features_dict))
                writer.write(example.SerializeToString())
                sample_num += 1
    
    writer.close()


