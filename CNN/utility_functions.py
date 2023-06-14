"""
Utility functions 

@author: Yuan Fang
"""
import os
import sys
import random

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import logging

from model_main import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Contribution score
def contribution_input_grad(inputs, model):
    """
    Calculate input x gradient
    """
    # inputs: Has the batch dimension
    seq = inputs
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([seq])
        prediction = model(seq)[0][0] 
    input_grad = tape.gradient(prediction, seq) * seq
    input_grad = tf.squeeze(input_grad, axis=0) 
    del tape
    return {'seq_grad':tf.reduce_sum(input_grad, axis=-1)}

def grad_4pos(inputs, model):
    """
    Calculate gradient w.r.t. input sequence
    """
    seq = inputs
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([seq])
        prediction = model(seq)[0][0] 
    input_grad = tape.gradient(prediction, seq) 
    input_grad = tf.squeeze(input_grad, axis=0) 
    del tape
    return {'seq_grad':input_grad.numpy()}

def extract_high_grad_region(grads, raw_grads, seqs, left_shift=50, right_shift=49, pool_bp=16):
    """
    Extract sequences of high gradient regions
    """
    high_grad_seqs = []
    high_grad_posis = []
    grads_list = []
    pooled_grads_list = []
    for i in range(len(grads)):
        # The two largest positive contribution score region
        for rank in [-1,-2]:
            rank_ind = np.argsort(grads[i])[rank]
            rank_posi = rank_ind * pool_bp + pool_bp // 2
            if grads[i][rank_ind] > 0:
                high_grad_posis.append(rank_ind)
                grads_list.append(raw_grads[i])
                pooled_grads_list.append(grads[i])
                
                if (rank_posi-left_shift) >= 0 and (rank_posi+1+right_shift) <= 2048:
                    high_grad_seqs.append(seqs[i][(rank_posi-left_shift):(rank_posi+1+right_shift)])
                    
                elif (rank_posi-left_shift) < 0:
                    high_grad_seqs.append(seqs[i][0:(rank_posi+1+right_shift)])
                    
                elif (rank_posi+1+right_shift) > 2048:
                    high_grad_seqs.append(seqs[i][(rank_posi-left_shift):2048])
            
        #if i%10==0: print(i)
    return high_grad_seqs, high_grad_posis, grads_list, pooled_grads_list

def extract_high_grad_region_loop(model, tfr_files, left_shift=50, right_shift=49, pool_bp=16):
    input_seqs = []
    contribution_score_list = []
    pooled_contribution_score_list = []
    
    for file in tfr_files:
        dset = tf.data.TFRecordDataset(file, compression_type='ZLIB').map(decode_fn_finetune)
        for i,sample in enumerate(dset):
            seq, gene_type = sample
            seq = tf.expand_dims(seq, axis=0)
            gene_type = tf.expand_dims(gene_type, axis=0)
            seq_string = ''
            for nuc in seq[0].numpy():
                if nuc[0] == 1: seq_string += 'A'
                elif nuc[1] == 1: seq_string += 'C'
                elif nuc[2] == 1: seq_string += 'G'
                elif nuc[3] == 1: seq_string += 'T'
                else: seq_string += 'N'
            input_seqs.append(seq_string)
            seq_grad = grad_4pos(seq, model)['seq_grad'] #[2048,4]
            seq_grad2 = contribution_input_grad(seq, model)
            contribution_score_list.append(seq_grad)
            pooled_contribution_scores = tf.nn.avg_pool1d(seq_grad2['seq_grad'][np.newaxis, :, np.newaxis], pool_bp, pool_bp, 'VALID')[0, :, 0].numpy()
            pooled_contribution_score_list.append(pooled_contribution_scores)
            
            #if i%10==0: print(i)
            
    high_grad_seqs, high_grad_posis, grads_list, pooled_grads_list = extract_high_grad_region(pooled_contribution_score_list, 
                                                                                              contribution_score_list,
                                                                                              input_seqs,
                                                                                              left_shift, 
                                                                                              right_shift, 
                                                                                              pool_bp)
        
    return (input_seqs,
            high_grad_seqs,
            high_grad_posis,
            grads_list,
            pooled_grads_list)


# Replacement
def modify_seq(seq, seq_pattern, replace_posi):
    """
    Modify the original sequence
    """
    new_seq = np.copy(seq)
    for i,nuc in enumerate(seq_pattern):
        if nuc=='A': new_seq[i+replace_posi]=np.array([1,0,0,0])
        elif nuc=='T': new_seq[i+replace_posi]=np.array([0,0,0,1])
        elif nuc=='G': new_seq[i+replace_posi]=np.array([0,0,1,0])
        elif nuc=='C': new_seq[i+replace_posi]=np.array([0,1,0,0])
        else: new_seq[i+replace_posi]=np.array([0.25,0.25,0.25,0.25])
    return new_seq

def create_modified_seq_tfrecord(tfr_files, output_dir, seq_pattern, replace_posi, tf_opts):
    """
    Write modified sequences into tfrecords
    """
    rand_seqs = []
    for file in tfr_files:
        dset = tf.data.TFRecordDataset(file, compression_type='ZLIB').map(decode_fn_finetune_2)
        for sample in dset:
            seq, gene_type = sample
            rand_seqs.append(seq.numpy())
        
    total_written = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir+'/rand_'+seq_pattern+'_'+str(replace_posi)+'.tfrecords'
    with tf.io.TFRecordWriter(output_file, tf_opts) as writer:
        for i in range(len(rand_seqs)):
            seq = rand_seqs[i] #[2048, 4]
            new_seq = modify_seq(seq, seq_pattern, replace_posi)
            target = np.array([1]) #[1,]
            
            new_seq = new_seq.astype('uint8')
            target = target.astype('uint8')
            features_dict = {
                'sequence': create_byte_feature(new_seq),
                'target': create_byte_feature(target)
                }

            # write example
            example = tf.train.Example(features=tf.train.Features(feature=features_dict))
            writer.write(example.SerializeToString())
            total_written += 1
    writer.close()
    print('The number of written samples is {0}'.format(total_written))
    return


    


