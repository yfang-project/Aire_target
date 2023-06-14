"""
Generate fine-tuning data

@author: Yuan Fang
"""
import os
import sys
import glob
import random 

import tensorflow as tf
tf.enable_eager_execution()

import pandas as pd
import numpy as np
from copy import deepcopy
import json
import pickle
from pathlib import Path
from kipoi.data import Dataset
from collections import OrderedDict
from pybedtools import Interval

from kipoi.metadata import GenomicRanges
from concise.utils.helper import get_from_module
from concise.preprocessing import encodeDNA
from genomelake.extractors import FastaExtractor, BigwigExtractor, ArrayExtractor
from kipoi_utils.data_utils import get_dataset_item
from kipoiseq.dataloaders.sequence import BedDataset
import gin
import logging

from model_main import *


class IntervalReader:
    def __init__(self, 
                 bed_file,
                 num_chr=False,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 chromosome_lens=None,
                 resize_width=None,
                 max_width=2048,
                 ):
        """
        Read and parse a .bed file
        
        Args:
          bed_file: a BED file
          num_chr: if True, remove the 'chr' prefix if existing in the chromosome names
          incl_chromosomes (list of str): list of chromosomes to keep.
          excl_chromosomes (list of str): list of chromosomes to exclude.
          chromosome_lens (dict of int): dictionary with chromosome lengths
          resize_width (int): desired interval width. The resize fixes the center
              of the interval.
              
        Reference: Avsec, Ž., Zeitlinger, J., et al., Nature Genetics, 2021
        """
        self.tsv_file = bed_file
        self.num_chr = num_chr
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.chromosome_lens = chromosome_lens
        self.resize_width = resize_width
        self.max_width = max_width
       

        columns = list(pd.read_csv(self.tsv_file, nrows=0, sep='\t').columns)

        if not columns[0].startswith("CHR") and not columns[0].startswith("#CHR"):
            self.columns = None
            skiprows = None
        else:
            # There exists a header
            self.columns = columns
            skiprows = [0]
            
        dtypes = {i: d for i, d in enumerate([str, int, int, str, str, str])}
        self.df = pd.read_csv(self.tsv_file,
                              header=None,
                              comment='#',
                              skiprows=skiprows,
                              dtype=dtypes,
                              sep='\t')
        if self.num_chr and self.df.iloc[0][0].startswith("chr"):
            self.df[0] = self.df[0].str.replace("^chr", "")
        if not self.num_chr and not self.df.iloc[0][0].startswith("chr"):
            self.df[0] = "chr" + self.df[0]

        # Omit data outside chromosomes
        if incl_chromosomes is not None:
            self.df = self.df[self.df[0].isin(incl_chromosomes)]
        if excl_chromosomes is not None:
            self.df = self.df[~self.df[0].isin(excl_chromosomes)]

        # Make the chromosome name a categorical variable
        self.df[0] = pd.Categorical(self.df[0])
        
        # Remove intervals longer than the max_width
        n_int = self.df.shape[0]
        seq_length = self.df[2] - self.df[1]
        valid_seqs = seq_length <= self.max_width
        self.df = self.df[valid_seqs]
        
        if len(self.df) != n_int:
            print(f"Skipped {n_int - len(self.df)} intervals longer than max input width {self.max_width}")
        
        # Skip intervals outside of the genome
        if (self.chromosome_lens is not None) and (resize_width is not None):
            n_int = self.df.shape[0]
            center = (self.df[1] + self.df[2]) // 2
            valid_seqs = ((center > self.resize_width // 2 + 1) &
                          (center < self.df[0].map(self.chromosome_lens).astype(int) - self.resize_width // 2 - 1))
            self.df = self.df[valid_seqs]

            if len(self.df) != n_int:
                print(f"Skipped {n_int - len(self.df)} intervals"
                      " outside of the genome size")

    def getIntervalType(self, idx):
        """
        Returns pybedtools.Interval and gene type
        """
        df_row = self.df.iloc[idx]
        seq_interval = Interval(df_row[0], df_row[1], df_row[2], strand=df_row[5])
        seq_type = df_row[3]
        
        return seq_interval, seq_type

    def df_len(self):
        return len(self.df)

    def shuffle_inplace(self):
        """
        Shuffle the interval
        """
        shuffled_idx = random.sample(list(range(self.df.shape[0])), self.df.shape[0])
        self.df = self.df.iloc[shuffled_idx]


def chrom_sizes(fasta_file):
    """
    Return chromosome sizes for a fasta file
    """
    from pysam import FastaFile
    fa = FastaFile(fasta_file)
    chrom_lens = OrderedDict([(name, l) for name, l in zip(fa.references, fa.lengths)])
    if len(chrom_lens) == 0:
        raise ValueError(f"no chromosomes found in fasta file: {fasta_file}. "
                         "Make sure the file path is correct and that the fasta index "
                         "file {fasta_file}.fai is up to date")
    fa.close()
    return chrom_lens

class FinetuneData():
    def __init__(self,
                 fa_path,
                 intervals_file,
                 peak_width=None,
                 pad_length=2048,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 num_chr=False,
                 interval_transformer=None,
                 shuffle=True):
        """
        Prepare single example of finetune data
        """
        self.fa_path = fa_path
        self.intervals_file = intervals_file
        self.peak_width = peak_width
        self.pad_length = pad_length
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.num_chr = num_chr
        self.interval_transformer = interval_transformer
        self.shuffle = shuffle
       
        # Not specified yet
        self.fasta_extractor = None
        
        # Load chromosome lengths
        self.chrom_lens = chrom_sizes(self.fa_path)
                
        self.tsv = IntervalReader(bed_file=self.intervals_file,
                                  num_chr=self.num_chr,
                                  incl_chromosomes=self.incl_chromosomes,
                                  excl_chromosomes=self.excl_chromosomes,
                                  chromosome_lens=self.chrom_lens,
                                  resize_width=self.peak_width,
                                  max_width=self.pad_length)
        
        if self.shuffle:
            self.tsv.shuffle_inplace()
        self.df = self.tsv.df
        
    def df_len(self):
        return len(self.df)

    def getitem(self, idx):
        # idx is the row index after removing invalid intervals
        if self.fasta_extractor is None:
            # if True, the extracted sequence is reverse complemented in case interval.strand == "-"
            self.fasta_extractor = FastaExtractor(self.fa_path, use_strand=True)

        interval,seq_type = self.tsv.getIntervalType(idx)

        # Transform the input interval
        if self.interval_transformer is not None:
            interval = self.interval_transformer(interval)

        # Extract and one-hot encode DNA sequence
        sequence = self.fasta_extractor([interval])[0] # np.array((interval_len, 4))
        
        inputs = {"seq": sequence}
        seq_type = 1 if seq_type=='Aire_induced_gene' else 0
        
        outputs = {"type": seq_type}
        
        out = {"inputs": inputs,
               "targets": outputs}

        return out

def create_input_tfrecord(metadata, output_file, bed_file, tf_opts):
    """
    Write finetune data to tfrecords
    """
    total_written = 0
    with tf.io.TFRecordWriter(output_file, tf_opts) as writer:
        for i in range(metadata.df.shape[0]):
            single_sample = metadata.getitem(i)
            seq = single_sample['inputs']['seq'] 
                
            target = single_sample['targets']['type'] 
            target = np.array([target]) 
            
            seq = seq.astype('uint8')
            target = target.astype('uint8')
            features_dict = {
                'sequence': create_byte_feature(seq),
                'target': create_byte_feature(target)
                }

            # write example
            example = tf.train.Example(features=tf.train.Features(feature=features_dict))
            writer.write(example.SerializeToString())
            total_written += 1
    writer.close()
    print('The number of written samples is {0}'.format(total_written))
    return


        
