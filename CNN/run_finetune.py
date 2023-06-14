"""
Fine-tuning for classifying AIGs vs ANGs

AIGs: Aire-induced genes
ANGs: Aire-neutral genes

@author: Yuan Fang
"""
import os
import sys
import random

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import logging

from model_main import *
#tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

mirrored_strategy = tf.distribute.MirroredStrategy()

print("tensorflow version", tf.__version__)
print("keras version", tf.keras.__version__)
print("GPU Available: ", tf.config.experimental.list_physical_devices('GPU'))
print("All Physical Devices", tf.config.experimental.list_physical_devices())

# Compute a global batch size using a number of replicas.
BATCH_SIZE_PER_REPLICA = int(sys.argv[1]) 
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     mirrored_strategy.num_replicas_in_sync)
print('Global batch size: ', global_batch_size)

# Create finetune dataset
class aire_dataset:
    def __init__(self, seq_len, seq_depth, 
                 batch_size, dset_buffer_size, tfr_path):
        self.seq_len = seq_len
        self.seq_depth = seq_depth
        self.batch_size = batch_size
        self.dset_buffer_size = dset_buffer_size
        self.tfr_path = tfr_path
        
    def generate_parser(self, raw=False):
        def decode_fn_finetune(record_bytes):
            features = {"sequence": tf.io.FixedLenFeature([], dtype=tf.string),
                        "target": tf.io.FixedLenFeature([], dtype=tf.string)
                        }
            parsed_example = tf.io.parse_single_example(record_bytes, features)
            
            # decode
            sequence = tf.io.decode_raw(parsed_example['sequence'], tf.uint8)
            gene_type = tf.io.decode_raw(parsed_example['target'], tf.uint8)
    
            if not raw:
                sequence = tf.reshape(sequence, [self.seq_len, self.seq_depth]) #[2048, 4]
                sequence = tf.cast(sequence, tf.float32)
                gene_type = tf.reshape(gene_type, []) 
                gene_type = tf.cast(gene_type, tf.int32)
            return sequence,gene_type
        return decode_fn_finetune
    
    
    def generate_dataset(self, cycle_length=4):
        tfr_files = os.listdir(self.tfr_path)
        tfr_files = [os.path.join(self.tfr_path, file) for file in tfr_files 
                     if file.endswith('.tfrecords')]
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
batch_size = global_batch_size
dset_buffer_size_train = 1000
dset_buffer_size_val = 500
dset_buffer_size_test = 200
AUTOTUNE = tf.data.experimental.AUTOTUNE
tfr_train_path = str(sys.argv[2])
tfr_test_path = str(sys.argv[3])
tfr_valid_path = str(sys.argv[4])


generate_train = aire_dataset(seq_len, seq_depth, 
                              batch_size, dset_buffer_size_train, 
                              tfr_train_path)                               
train_data = generate_train.generate_dataset()


generate_val = aire_dataset(seq_len, seq_depth, 
                                batch_size, dset_buffer_size_val, 
                                tfr_valid_path)   
val_data = generate_val.generate_dataset()


generate_test = aire_dataset(seq_len, seq_depth, 
                                 batch_size, dset_buffer_size_test, 
                                 tfr_test_path)   
test_data = generate_test.generate_dataset()

print("train_data", train_data)
print("validation_data", val_data)
print("test_data", test_data)

# Pretrained model parameters 
conv_act='gelu'
head_act='softplus'
num_targets = 1643  
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


# Finetune stage 1 optimizer parameters
lr = 1e-5
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07
pretrained_path = str(sys.argv[5])
pretrained_prefix = str(sys.argv[6])

# Finetune stage 1: freeze the main body
with mirrored_strategy.scope():
    pre_trained_model = SeqPretrainModel(conv_act, head_act, num_targets,
                                         pool_size, conv_tower_layer, dilated_conv_tower_layer,
                                         conv_filters, conv_filter_ratios, conv1_kernel_size,
                                         pointwise_kernal_size, conv_tower_kernal_size, dilated_conv_tower_kernal_size,
                                         dilated_dropout_rate, pointwise_dropout_rate, padding,
                                         batchnorm, conv_tower_pointwise, dilated_conv_tower_pointwise,
                                         dilated_dropout, pointwise_dropout, return_embeddings)
    
    dummy_train_x = tf.random.uniform(shape=[1,2048,4], maxval=2, dtype=tf.float32, seed=10)
    dummy_output = pre_trained_model(dummy_train_x)
    print(pre_trained_model.summary())
    
    modelPath = pretrained_path
    pre_trained_model.load_weights(os.path.join(modelPath, pretrained_prefix))
    
    finetune_model = SeqFinetuneModel(pre_trained_model,
                                      dummy_train_x,
                                      conv_act='gelu',
                                      p_head_act='softplus',
                                      i_head_act='sigmoid',
                                      num_targets=1,
                                      conv1_num=10,
                                      pool_size=2,
                                      conv_tower_layer=3,
                                      dilated_conv_tower_layer=5,
                                      conv_filters=768,
                                      imb_filter=3,
                                      conv_filter_ratios=[0.5, 0.65, 0.85, 1],
                                      conv1_kernel_size=20,
                                      pointwise_kernal_size=1,
                                      conv_tower_kernal_size=5,
                                      dilated_conv_tower_kernal_size=3,
                                      dilated_dropout_rate=0.3,
                                      pointwise_dropout_rate=0.05,
                                      padding='same',
                                      batchnorm=True,
                                      conv_tower_pointwise=False,
                                      dilated_conv_tower_pointwise=True,
                                      dilated_dropout=True,
                                      pointwise_dropout=True,
                                      return_embeddings=False)
    dummy_train_x_seq = tf.random.uniform(shape=[1,2048,4], maxval=2, dtype=tf.float32, seed=10)
    dummy_output = finetune_model(dummy_train_x_seq)
    print(finetune_model.summary())
    
     # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr,
                                          beta_1 = beta_1,
                                          beta_2 = beta_2,
                                          epsilon = epsilon)
    
    # Compile
    finetune_model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                           optimizer = optimizer,
                           metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall(),
                                      F1_Score()])  
    
# Run finetune stage 1
outputFolder = './finetune_stage1_models'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/model-{epoch:02d}-{val_f1_score:.4f}"

BestFolder = './finetune_stage1_model_best'
if not os.path.exists(BestFolder):
    os.makedirs(BestFolder)
bestfilepath=BestFolder+"/best_model-{epoch:02d}-{val_f1_score:.4f}"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                         monitor='val_f1_score', 
                                                         mode='max',
                                                         verbose=1,
                                                         save_best_only=False, 
                                                         save_weights_only=False,
                                                         save_freq='epoch')

early_stop = EarlyStoppingMin(monitor='val_f1_score', 
                              mode='max',
                              verbose=1,
                              patience=20, 
                              restore_best_weights=True,
                              min_epoch=10)

save_best = tf.keras.callbacks.ModelCheckpoint(bestfilepath,
                                               monitor='val_f1_score', 
                                               mode='max',
                                               verbose=1,
                                               save_best_only=True, 
                                               save_weights_only=False)

csv_logger = tf.keras.callbacks.CSVLogger('finetune_stage1_log.txt', separator="\t", append=True)

train_sequence_num = int(sys.argv[7])
train_steps_per_epoch = int(train_sequence_num/global_batch_size)
valid_sequence_num = int(sys.argv[8])
valid_steps_per_epoch = int(valid_sequence_num/global_batch_size)
training_history = finetune_model.fit(train_data,
                                      validation_data = val_data,
                                      validation_steps = valid_steps_per_epoch,
                                      validation_freq = 1,
                                      callbacks = [checkpoint_callback, csv_logger, early_stop, save_best],
                                      steps_per_epoch = train_steps_per_epoch, 
                                      epochs = 100,
                                      verbose = 1)

hist_df = pd.DataFrame(training_history.history) 
hist_csv_file = 'finetune_stage1_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

outputFianlPath = './finetune_stage1_model_final'
if not os.path.exists(outputFianlPath):
    os.makedirs(outputFianlPath)
finetune_model.save_weights(os.path.join(outputFianlPath, 'model_final_weights'))


# Evaluate on test set
test_sequence_num = int(sys.argv[9])
test_steps_per_epoch = int(test_sequence_num/global_batch_size)
test_eval = finetune_model.evaluate(test_data,
                                    verbose = 1,
                                    steps = test_steps_per_epoch)
test_eval = dict(zip(finetune_model.metrics_names, test_eval))

test_df = pd.DataFrame.from_dict([test_eval]) 
test_df.to_csv('finetune_stage1_test_evaluations.txt', sep='\t', index=False)

   
# Finetune stage 2 optimizer parameters
lr = 1e-7
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07
finetuned_path = str(sys.argv[10])
finetuned_prefix = str(sys.argv[11])  


# Finetune stage 2: unfreeze the pointwise layer of the main body
with mirrored_strategy.scope():
    finetune_model = AireFinetuneModel(conv_act='gelu',
                                        p_head_act='softplus',
                                        i_head_act='sigmoid',
                                        num_targets=1,
                                        conv1_num=10,
                                        pool_size=2,
                                        conv_tower_layer=3,
                                        dilated_conv_tower_layer=5,
                                        conv_filters=768,
                                        imb_filter=3,
                                        conv_filter_ratios=[0.5, 0.65, 0.85, 1],
                                        conv1_kernel_size=20,
                                        pointwise_kernal_size=1,
                                        conv_tower_kernal_size=5,
                                        dilated_conv_tower_kernal_size=3,
                                        dilated_dropout_rate=0.3,
                                        pointwise_dropout_rate=0.05,
                                        padding='same',
                                        batchnorm=True,
                                        conv_tower_pointwise=False,
                                        dilated_conv_tower_pointwise=True,
                                        dilated_dropout=True,
                                        pointwise_dropout=True,
                                        return_embeddings=False)
    dummy_train_x_seq = tf.random.uniform(shape=[1,2048,4], maxval=2, dtype=tf.float32, seed=10)
    dummy_output = finetune_model(dummy_train_x_seq)
    print(finetune_model.summary())
    
    modelPath = finetuned_path
    finetune_model.load_weights(os.path.join(modelPath, finetuned_prefix))
    
    finetune_model.body.trainable = True
    finetune_model.body.conv1.trainable = False
    for l in finetune_model.body.conv_tower.layers:
        l.trainable = False
    for l in finetune_model.body.dilated_conv_tower.layers:
        l.trainable = False
    finetune_model.body.pointwise_layer.trainable = True
    print(finetune_model.summary())
    
     # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr,
                                          beta_1 = beta_1,
                                          beta_2 = beta_2,
                                          epsilon = epsilon)
    
    # Compile
    finetune_model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                           optimizer = optimizer,
                           metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall(),
                                      F1_Score()])  

# Run finetune stage 2
outputFolder = './finetune_stage2_models'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/model-{epoch:02d}-{val_f1_score:.4f}"

BestFolder = './finetune_stage2_model_best'
if not os.path.exists(BestFolder):
    os.makedirs(BestFolder)
bestfilepath=BestFolder+"/best_model-{epoch:02d}-{val_f1_score:.4f}"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                         monitor='val_f1_score', 
                                                         mode='max',
                                                         verbose=1,
                                                         save_best_only=False, 
                                                         save_weights_only=False,
                                                         save_freq='epoch')

early_stop = EarlyStoppingMin(monitor='val_f1_score', 
                              mode='max',
                              verbose=1,
                              patience=5, 
                              restore_best_weights=True,
                              min_epoch=5)

save_best = tf.keras.callbacks.ModelCheckpoint(bestfilepath,
                                               monitor='val_f1_score', 
                                               mode='max',
                                               verbose=1,
                                               save_best_only=True, 
                                               save_weights_only=False)

csv_logger = tf.keras.callbacks.CSVLogger('finetune_stage2_log.txt', separator="\t", append=True)


training_history = finetune_model.fit(train_data,
                                      validation_data = val_data,
                                      validation_steps = valid_steps_per_epoch,
                                      validation_freq = 1,
                                      callbacks = [checkpoint_callback, early_stop, csv_logger, save_best],
                                      steps_per_epoch = train_steps_per_epoch, 
                                      epochs = 10,
                                      verbose = 1)

hist_df = pd.DataFrame(training_history.history) 
hist_csv_file = 'finetune_stage2_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

outputFianlPath = './finetune_stage2_model_final'
if not os.path.exists(outputFianlPath):
    os.makedirs(outputFianlPath)
finetune_model.save_weights(os.path.join(outputFianlPath, 'model_final_weights'))


# Evaluate on test set
test_eval = finetune_model.evaluate(test_data,
                                    verbose = 1,
                                    steps = test_steps_per_epoch)
test_eval = dict(zip(finetune_model.metrics_names, test_eval))

test_df = pd.DataFrame.from_dict([test_eval]) 
test_df.to_csv('finetune_stage2_test_evaluations.txt', sep='\t', index=False)


