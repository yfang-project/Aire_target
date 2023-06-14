"""
Model pre-training using modifed Basenji2 data

@author: Yuan Fang
"""
import os
import sys
import random

import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import logging

from model_main import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


mirrored_strategy = tf.distribute.MirroredStrategy()

print("tensorflow version", tf.__version__)
print("keras version", tf.keras.__version__)
print("GPU Available: ", tf.config.experimental.list_physical_devices('GPU'))
print("All Physical Devices", tf.config.experimental.list_physical_devices())

# Compute a global batch size using a number of replicas.
BATCH_SIZE_PER_REPLICA = 64
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     mirrored_strategy.num_replicas_in_sync)
print('Global batch size: ', global_batch_size)


# Create pretrain dataset
class pretrain_dataset:
    def __init__(self, seq_len, seq_depth, 
                 target_len, num_targets, 
                 prefix, batch_size, dset_buffer_size, tfr_path):
        self.seq_len = seq_len
        self.seq_depth = seq_depth
        self.target_len = target_len
        self.num_targets = num_targets
        self.prefix = prefix
        self.batch_size = batch_size
        self.dset_buffer_size = dset_buffer_size
        self.tfr_path = tfr_path
        
    def generate_parser(self, raw=False):
        def decode_fn_pretrain(record_bytes):
            features = {
                "sequence": tf.io.FixedLenFeature([], dtype=tf.string),
                "target": tf.io.FixedLenFeature([], dtype=tf.string)
                }
            parsed_example = tf.io.parse_single_example(record_bytes, features)
            
            # decode sequence
            seq = tf.io.decode_raw(parsed_example['sequence'], tf.uint8)
            if not raw:
                seq = tf.reshape(seq, [self.seq_len, self.seq_depth])
                seq = tf.cast(seq, tf.float32)
        
            # decode targets
            targets = tf.io.decode_raw(parsed_example['target'], tf.float16)
            if not raw:
              targets = tf.reshape(targets, [self.target_len, self.num_targets])
              targets = tf.cast(targets, tf.float32)
                
            return seq, targets
        return decode_fn_pretrain
    
    
    def generate_dataset(self, cycle_length=4):
        tfr_files = os.listdir(self.tfr_path)
        tfr_files = [os.path.join(self.tfr_path, file) for file in tfr_files 
                     if (file.endswith('.tfrecords') and file.startswith(self.prefix))]
        tfr_files.sort()
        
        d = tf.data.Dataset.from_tensor_slices(tf.constant(tfr_files))
        d = d.repeat()
        d = d.shuffle(buffer_size=len(tfr_files))
        d = d.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, 
                                                     compression_type='ZLIB').map(self.generate_parser(), 
                                                                    num_parallel_calls=AUTOTUNE),
            cycle_length=cycle_length,
            num_parallel_calls=AUTOTUNE)
        d = d.shuffle(buffer_size=self.dset_buffer_size, reshuffle_each_iteration=True)
        d = d.batch(self.batch_size)
        d = d.prefetch(buffer_size=AUTOTUNE)
        return d



seq_len = 2048 # Input sequence length
seq_depth = 4 # One-hot encoding 
target_len = 16
num_targets = 1643  # Sequencing track number
batch_size = global_batch_size
dset_buffer_size = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE
tfr_path = str(sys.argv[1])

generate_train = pretrain_dataset(seq_len, seq_depth, 
                                  target_len, num_targets, 
                                  'train', batch_size, dset_buffer_size, 
                                  tfr_path)

train_data = generate_train.generate_dataset()


generate_val = pretrain_dataset(seq_len, seq_depth, 
                                target_len, num_targets, 
                                'valid', batch_size, dset_buffer_size, 
                                tfr_path)

val_data = generate_val.generate_dataset()

print("train_data", train_data)
print("validation_data", val_data)


# Model parameters 
conv_act='gelu'
head_act='softplus'
pool_size=2
conv_tower_layer=3
dilated_conv_tower_layer=5
conv_filters=768
conv_filter_ratios=[0.5, 0.65, 0.85, 1]
conv1_kernel_size=20
pointwise_kernal_size=8
conv_tower_kernal_size=5
dilated_conv_tower_kernal_size=3
dilated_dropout_rate=0.3
pointwise_dropout_rate=0.05
padding='same'
batchnorm=True
conv_tower_pointwise=False
dilated_conv_tower_pointwise=True
dilated_dropout=True
pointwise_dropout=True
return_embeddings=False


# Optimizer parameters
warmup_steps = 5000
initial_learning_rate = 1e-4
decay_schedule = initial_learning_rate
beta_1 = 0.9
beta_2 = 0.999
clip_norm = 2
epsilon = 1e-07

# Loss
loss_id = int(sys.argv[2])
total_weight = float(sys.argv[3])
pretrain_loss = decode_loss(loss_id, total_weight)

with mirrored_strategy.scope():
    model = SeqPretrainModel(conv_act, head_act, num_targets,
                              pool_size, conv_tower_layer, dilated_conv_tower_layer,
                              conv_filters, conv_filter_ratios, conv1_kernel_size,
                              pointwise_kernal_size, conv_tower_kernal_size, dilated_conv_tower_kernal_size,
                              dilated_dropout_rate, pointwise_dropout_rate, padding,
                              batchnorm, conv_tower_pointwise, dilated_conv_tower_pointwise,
                              dilated_dropout, pointwise_dropout, return_embeddings)
    
    # Optimizer
    lr = WarmUp(warmup_steps, decay_schedule, initial_learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr,
                                          beta_1 = beta_1,
                                          beta_2 = beta_2,
                                          clipnorm = clip_norm,
                                          epsilon = epsilon)
    
    # Compile
    model.compile(loss = pretrain_loss,
                  optimizer = optimizer,
                  metrics = [PearsonR(num_targets), R2(num_targets)])


# Train model
outputFolder = './pretrain_models'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/model-{epoch:02d}-{val_pearsonr:.4f}"

BestFolder = './pretrain_model_best'
if not os.path.exists(BestFolder):
    os.makedirs(BestFolder)
bestfilepath=BestFolder+"/best_model-{epoch:02d}-{val_pearsonr:.4f}"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                         monitor='val_pearsonr', 
                                                         mode='max',
                                                         verbose=1,
                                                         save_best_only=False, 
                                                         save_weights_only=False,
                                                         save_freq='epoch')

early_stop = EarlyStoppingMin(monitor='val_pearsonr', 
                              mode='max', 
                              verbose=1,
                              patience=15, 
                              restore_best_weights=True,
                              min_epoch=5)

save_best = tf.keras.callbacks.ModelCheckpoint(bestfilepath,
                                               monitor='val_pearsonr', 
                                               mode='max',
                                               verbose=1,
                                               save_best_only=True, 
                                               save_weights_only=False)

csv_logger = tf.keras.callbacks.CSVLogger('pretrain_log.txt', separator="\t", append=True)


training_history = model.fit(train_data,
                             validation_data = val_data,
                             validation_steps = 80,
                             validation_freq = 1,
                             callbacks = [checkpoint_callback, csv_logger, early_stop, save_best],
                             steps_per_epoch = 1000,
                             epochs = 100,
                             verbose = 1)

hist_df = pd.DataFrame(training_history.history) 
hist_csv_file = 'training_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

outputFianlPath = './pretrain_model_final'
if not os.path.exists(outputFianlPath):
    os.makedirs(outputFianlPath)
model.save_weights(os.path.join(outputFianlPath, 'model_final_weights'))

print(model.summary())

# Evaluate on test set
generate_test = pretrain_dataset(seq_len, seq_depth, 
                                 target_len, num_targets, 
                                 'test', batch_size, dset_buffer_size, 
                                 tfr_path)
test_data = generate_test.generate_dataset()

    
test_eval = model.evaluate(test_data,
                           verbose = 1,
                           steps = 75)
test_eval = dict(zip(model.metrics_names, test_eval))

test_df = pd.DataFrame.from_dict([test_eval]) 
test_df.to_csv('test_evaluations.txt', sep='\t', index=False)



