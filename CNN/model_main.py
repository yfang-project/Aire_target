"""
Metrics, losses, layers, models and utility functions
"""
import os
import sys
import random

import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

######################## Metrics and losses ########################

class PearsonR(tf.keras.metrics.Metric):
  '''
  Reference: Basenji metrics on GitHub
  '''
  def __init__(self, num_targets, summarize=True, name='pearsonr', **kwargs):
    super().__init__(name=name, **kwargs)
    
    self._summarize = summarize
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')
    self._pred_sum = self.add_weight(name='pred_sum', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    if len(y_true.shape) == 2:
      reduce_axes = 0
    else:
      reduce_axes = [0,1]

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
    self._product.assign_add(product)

    true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
    self._true_sumsq.assign_add(true_sumsq)

    pred_sum = tf.reduce_sum(y_pred, axis=reduce_axes)
    self._pred_sum.assign_add(pred_sum)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
    self._pred_sumsq.assign_add(pred_sumsq)

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=reduce_axes)
    self._count.assign_add(count)

  def result(self):
    true_mean = tf.divide(self._true_sum, self._count)
    true_mean2 = tf.math.square(true_mean)
    pred_mean = tf.divide(self._pred_sum, self._count)
    pred_mean2 = tf.math.square(pred_mean)

    term1 = self._product
    term2 = -tf.multiply(true_mean, self._pred_sum)
    term3 = -tf.multiply(pred_mean, self._true_sum)
    term4 = tf.multiply(self._count, tf.multiply(true_mean, pred_mean))
    
    covariance = term1 + term2 + term3 + term4

    true_var = self._true_sumsq - tf.multiply(self._count, true_mean2)
    
    pred_var = self._pred_sumsq - tf.multiply(self._count, pred_mean2)
    
    pred_var = tf.where(tf.greater(pred_var, 1e-12),
                        pred_var,
                        np.inf*tf.ones_like(pred_var))
    
    tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
    
    correlation = tf.divide(covariance, tp_var)

    if self._summarize:
        return tf.reduce_mean(correlation)
    else:
        return correlation

  def reset_states(self):
      K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])
      
class R2(tf.keras.metrics.Metric):
  '''
  Reference: Basenji metrics on GitHub
  '''
  def __init__(self, num_targets, summarize=True, name='r2', **kwargs):
    super().__init__(name=name, **kwargs)
    
    self._summarize = summarize
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    if len(y_true.shape) == 2:
      reduce_axes = 0
    else:
      reduce_axes = [0,1]

    true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
    self._true_sumsq.assign_add(true_sumsq)

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
    self._product.assign_add(product)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
    self._pred_sumsq.assign_add(pred_sumsq)

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=reduce_axes)
    self._count.assign_add(count)

  def result(self):
    true_mean = tf.divide(self._true_sum, self._count)
    true_mean2 = tf.math.square(true_mean)

    total = self._true_sumsq - tf.multiply(self._count, true_mean2)

    resid1 = self._pred_sumsq
    resid2 = -2*self._product
    resid3 = self._true_sumsq
    resid = resid1 + resid2 + resid3

    r2 = tf.ones_like(self._shape, dtype=tf.float32) - tf.divide(resid, total)

    if self._summarize:
        return tf.reduce_mean(r2)
    else:
        return r2

  def reset_states(self):
    K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])
    

class F1_Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)
        
def poisson_multinomial(y_true, y_pred, total_weight=1, epsilon=1e-6, rescale=True):
    seq_len = tf.cast(y_true.shape[1], dtype=tf.float32) #128
    
    # add epsilon to protect against tiny values
    y_true += epsilon #[B, L, T]
    y_pred += epsilon #[B, L, T]

    # sum across lengths
    s_true = tf.math.reduce_sum(y_true, axis=-2, keepdims=True) #[B, 1, T]
    s_pred = tf.math.reduce_sum(y_pred, axis=-2, keepdims=True) #[B, 1, T]

    # normalize to sum to one
    p_pred = y_pred / s_pred #[B, L, T]
    
    # total count poisson loss
    s_true_t = tf.transpose(s_true, [0, 2, 1])
    s_pred_t = tf.transpose(s_pred, [0, 2, 1])
    poisson_term = tf.keras.losses.poisson(s_true_t, s_pred_t) #[B, T]
    poisson_term /= tf.cast(seq_len, dtype=tf.float32)
    
    # multinomial loss
    pl_pred = tf.math.log(p_pred) #[B, L, T]
    multinomial_dot = -tf.math.multiply(y_true, pl_pred) #[B, L, T]
    multinomial_term = tf.math.reduce_sum(multinomial_dot, axis=-2) #[B, T]
    multinomial_term /= tf.cast(seq_len, dtype=tf.float32)
  
    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * poisson_term
    if rescale:
        loss_rescale = loss_raw*2/(1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale

class PoissonMultinomialPretrain(LossFunctionWrapper):
    def __init__(self, total_weight=1, reduction=losses_utils.ReductionV2.AUTO, name='poisson_multinomial_pretrain'):
        self.total_weight = total_weight
        pois_mn = lambda yt, yp: poisson_multinomial(yt, yp, self.total_weight)
        super(PoissonMultinomialPretrain, self).__init__(pois_mn, name=name, reduction=reduction)
        
def mse_multinomial(y_true, y_pred, total_weight=1, epsilon=1e-6, rescale=True):
    seq_len = tf.cast(y_true.shape[1], dtype=tf.float32) #L
    
    # add epsilon to protect against tiny values
    y_true += epsilon #[B, L, T]
    y_pred += epsilon #[B, L, T]

    # sum across lengths
    s_true = tf.math.reduce_sum(y_true, axis=-2, keepdims=True) #[B, 1, T]
    s_pred = tf.math.reduce_sum(y_pred, axis=-2, keepdims=True) #[B, 1, T]

    # normalize to sum to one
    p_pred = y_pred / s_pred #[B, L, T]
    
    # total count poisson loss
    s_true_t = tf.transpose(s_true, [0, 2, 1])
    s_pred_t = tf.transpose(s_pred, [0, 2, 1])
    mse_term = tf.keras.losses.mean_squared_error(tf.math.log(s_true_t), tf.math.log(s_pred_t)) #[B, T]
    mse_term /= tf.cast(seq_len, dtype=tf.float32)
    
    # multinomial loss
    pl_pred = tf.math.log(p_pred) #[B, L, T]
    multinomial_dot = -tf.math.multiply(y_true, pl_pred) #[B, L, T]
    multinomial_term = tf.math.reduce_sum(multinomial_dot, axis=-2) #[B, T]
    multinomial_term /= tf.cast(seq_len, dtype=tf.float32)
    
  
    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * mse_term
    if rescale:
        loss_rescale = loss_raw*2/(1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale

class MSEMultinomialPretrain(LossFunctionWrapper):
    def __init__(self, total_weight=1, reduction=losses_utils.ReductionV2.AUTO, name='mse_multinomial_pretrain'):
        self.total_weight = total_weight
        mse_mn = lambda yt, yp: mse_multinomial(yt, yp, self.total_weight)
        super(MSEMultinomialPretrain, self).__init__(mse_mn, name=name, reduction=reduction)
        
class GELU(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(GELU, self).__init__(**kwargs)
  def call(self, x):
    return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

class Softplus(tf.keras.layers.Layer):
  def __init__(self, exp_max=10000):
    super(Softplus, self).__init__()
    self.exp_max = exp_max
  def call(self, x):
    x = tf.clip_by_value(x, -self.exp_max, self.exp_max)
    return tf.keras.activations.softplus(x)
  def get_config(self):
    config = super().get_config().copy()
    config['exp_max'] = self.exp_max
    return config


######################## Layers and models ########################

class Conv1DTower(tf.keras.layers.Layer):
    def __init__(self, 
                 conv_filter_ratio,
                 current_layer,
                 conv_act='gelu',
                 filter_width=5,
                 pool_size = 2,
                 conv_filters=768,
                 padding='same',
                 batchnorm=True,
                 pointwise=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.filter_width = filter_width
        self.pool_size = pool_size
        self.padding = padding
        self.batchnorm = batchnorm
        self.pointwise=pointwise
        
        if self.batchnorm:
            self.bn_layer = tf.keras.layers.BatchNormalization(name=f"BN_tower_{current_layer}")
        self.conv = tf.keras.layers.Conv1D(int(conv_filters*conv_filter_ratio), 
                                           self.filter_width, 
                                           padding=self.padding,
                                           name=f"conv1d_tower_{current_layer}") 
       
        self.conv_act_fn = get_activation(conv_act)
        if self.pointwise:
            self.conv_point = tf.keras.layers.Conv1D(int(conv_filters*conv_filter_ratio), 
                                                     1, 
                                                     padding=self.padding,
                                                     name=f"pointwise_tower_{current_layer}") 
            self.skip_add = tf.keras.layers.Add(name=f"add_tower_{current_layer}")
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, 
                                                  strides=self.pool_size, 
                                                  padding=self.padding,
                                                  name=f"maxpool1d_tower_{current_layer}")
        
                          
    def call(self, x, training=False):
        if self.batchnorm:
            x = self.bn_layer(x, training=training)
        x = self.conv(x)
        x = self.conv_act_fn(x)
        if self.pointwise:
            x_before_pointwise = x
            x_after_pointwise = self.conv_point(x_before_pointwise)
            x = self.skip_add([x_before_pointwise, x_after_pointwise])
        x = self.max_pool(x)
        
        return x
                
            
class DilatedConv1DTower(tf.keras.layers.Layer):
    def __init__(self, 
                 current_layer,
                 conv_act='gelu',
                 filter_width=3,
                 conv_filters=768,
                 dilated_dropout_rate=0.3,
                 padding='same',
                 batchnorm=True,
                 pointwise=True,
                 dropout=True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.filter_width = filter_width
        self.padding = padding
        self.batchnorm = batchnorm
        self.pointwise=pointwise
        self.dropout = dropout
        
        if self.batchnorm:
            self.bn_layer1 = tf.keras.layers.BatchNormalization(name=f"BN_dilated_{current_layer}_1")
        self.dilated_conv = tf.keras.layers.Conv1D(conv_filters,
                                                   self.filter_width, 
                                                   padding=self.padding,
                                                   dilation_rate=2**current_layer,
                                                   name=f"conv1d_dilated_{current_layer}") 
       
        self.conv_act_fn1 = get_activation(conv_act)
        if self.pointwise:
            if self.batchnorm:
                self.bn_layer2 = tf.keras.layers.BatchNormalization(name=f"BN_dilated_{current_layer}_2")
            self.conv_point = tf.keras.layers.Conv1D(conv_filters,
                                                     1, 
                                                     padding=self.padding,
                                                     name=f"pointwise_dilated_{current_layer}") 
            self.conv_act_fn2 = get_activation(conv_act)
        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(rate=dilated_dropout_rate)
        self.skip_add = tf.keras.layers.Add(name=f"add_dilated_{current_layer}")
        
                          
    def call(self, x, training=False):
        prev_x = x
        if self.batchnorm:
            x = self.bn_layer1(prev_x, training=training)
        x = self.dilated_conv(x)
        x = self.conv_act_fn1(x)
        if self.pointwise:
            if self.batchnorm:
                x = self.bn_layer2(x, training=training)
            x = self.conv_point(x)
            x = self.conv_act_fn2(x)
        if self.dropout:
            x = self.dropout_layer(x, training=training)
        x = self.skip_add([prev_x, x])
        
        return x
            
class PointwiseLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 conv_act='gelu',
                 conv_filters=768,
                 padding='same',
                 dropout=True,
                 dropout_rate=0.05,
                 batchnorm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.padding = padding
       
        if self.batchnorm:
            self.bn_layer = tf.keras.layers.BatchNormalization(name="BN_pointwise")
        self.conv_point = tf.keras.layers.Conv1D(conv_filters*2,
                                                 1, 
                                                 padding=self.padding,
                                                 name="pointwise_layer") 
           
        self.conv_act_fn = get_activation(conv_act)
        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
    
    def call(self, x, training=False):
        if self.batchnorm:
            x = self.bn_layer(x, training=training)
        x = self.conv_point(x)
        x = self.conv_act_fn(x)
        if self.dropout:
            x = self.dropout_layer(x, training=training)
        return x    
    
class Conv1DDilatedConv1D(tf.keras.layers.Layer):
    """
    Conv and dilated convolutional layers
    """
    
    def __init__(self, 
                 conv_act='gelu',
                 pool_size=2,
                 conv_tower_layer=3,
                 dilated_conv_tower_layer=5,
                 conv_filters=768,
                 conv_filter_ratios=[0.5, 0.65, 0.85, 1],
                 conv1_kernel_size=20,
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
                 **kwargs):
        super().__init__(**kwargs)
        
        self.pool_size = pool_size
        self.conv_tower_layer = conv_tower_layer
        self.dilated_conv_tower_layer = dilated_conv_tower_layer
        self.padding = padding
        self.batchnorm = batchnorm
        self.conv_tower_pointwise = conv_tower_pointwise
        self.dilated_conv_tower_pointwise = dilated_conv_tower_pointwise
        self.dilated_dropout = dilated_dropout
        self.pointwise_dropout = pointwise_dropout
        
        self.conv1 = tf.keras.layers.Conv1D(int(conv_filters*conv_filter_ratios[0]), 
                                             conv1_kernel_size, 
                                             padding=self.padding,
                                             name="conv1d_1")

        self.conv_act_fn = get_activation(conv_act)
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, 
                                                  strides=self.pool_size, 
                                                  padding=self.padding,
                                                  name="maxpool1d_1") 
                         
        self.conv_tower = [Conv1DTower(conv_filter_ratio=conv_filter_ratios[i+1],
                                       current_layer=(i+1),
                                       conv_act=conv_act,
                                       filter_width=conv_tower_kernal_size,
                                       pool_size=self.pool_size,
                                       conv_filters=conv_filters,
                                       padding=self.padding,
                                       batchnorm=self.batchnorm,
                                       pointwise=self.conv_tower_pointwise,
                                       name=f'ConvTower_layer_{i+1}') 
                           for i in range(self.conv_tower_layer)]
        
        self.dilated_conv_tower = [DilatedConv1DTower(current_layer=(i+1),
                                                      conv_act=conv_act,
                                                      filter_width=dilated_conv_tower_kernal_size,
                                                      conv_filters=conv_filters,
                                                      dilated_dropout_rate=dilated_dropout_rate,
                                                      padding=self.padding,
                                                      batchnorm=self.batchnorm,
                                                      pointwise=self.dilated_conv_tower_pointwise,
                                                      dropout=self.dilated_dropout,
                                                      name=f'DilatedConvTower_layer_{i+1}') 
                                   for i in range(self.dilated_conv_tower_layer)]
        
        self.pointwise_layer = PointwiseLayer(conv_act=conv_act,
                                              conv_filters=conv_filters,
                                              padding=self.padding,
                                              dropout=self.pointwise_dropout,
                                              dropout_rate=pointwise_dropout_rate,
                                              batchnorm=self.batchnorm,
                                              name='pointwise_layer')
        

    def call(self, inp, training=False):
        # inp [B, L, 4]
        
        # Conv1D stem
        first_conv = self.conv1(inp)
        first_conv = self.conv_act_fn(first_conv)
        first_conv = self.max_pool(first_conv)
        
        # Conv1D tower
        conv_tower_inp = first_conv
        for i in range(self.conv_tower_layer):
            x = self.conv_tower[i](conv_tower_inp, training=training)
            conv_tower_inp = x
        
        # Dilated Conv1D tower
        dilated_conv_tower_inp = conv_tower_inp
        for i in range(self.dilated_conv_tower_layer):
            x = self.dilated_conv_tower[i](dilated_conv_tower_inp, training=training)
            dilated_conv_tower_inp = x
        
        # Pointwise Conv1D
        pointwise_output = self.pointwise_layer(dilated_conv_tower_inp, training=training)
        return pointwise_output
    
        
        
class PretrainHead(tf.keras.layers.Layer):
    def __init__(self, 
                 head_act='softplus',
                 num_targets=1643,
                 conv_kernel_size=8,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_targets = num_targets
        self.conv = tf.keras.layers.Conv1D(num_targets, 
                                           conv_kernel_size, 
                                           strides=conv_kernel_size,
                                           padding='valid',
                                           name="pre_train_head")
        self.conv_act_fn = get_activation(head_act)
       
    def call(self, body_output, training=False):
        x = self.conv(body_output)
        x = self.conv_act_fn(x)
        return x
        
class AireHead(tf.keras.layers.Layer):
    def __init__(self, 
                 conv_act='gelu',
                 i_head_act='sigmoid',
                 conv1_num=10,
                 dropout_rate=0.5,
                 conv_kernel_size=1,
                 imb_filter=3,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.conv1_num = conv1_num
        self.imb_filter = imb_filter
        self.conv = tf.keras.layers.Conv1D(conv1_num, 
                                           conv_kernel_size,
                                           strides=conv_kernel_size,
                                           padding='valid',
                                           name="aire_conv1d")
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.conv_act_fn = get_activation(conv_act)
        self.flatten = tf.keras.layers.Flatten(name="aire_flatten")
        self.dense1 = tf.keras.layers.Dense(imb_filter, use_bias=True, 
                                            name="aire_dense1")
        self.dense_act_fn = get_activation(conv_act)
        self.dense2 = tf.keras.layers.Dense(1, activation=i_head_act, use_bias=True,
                                            name="aire_dense2")
        
    def call(self, body_output_aire, training=False):
        x = self.conv(body_output_aire)
        x = self.dropout_layer(x, training=training)
        x = self.conv_act_fn(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense_act_fn(x)
        x = self.dense2(x)
        return x
    
class SeqPretrainModel(tf.keras.Model):
    def __init__(self, 
                 conv_act='gelu',
                 head_act='softplus',
                 num_targets=1643,
                 pool_size=2,
                 conv_tower_layer=3,
                 dilated_conv_tower_layer=5,
                 conv_filters=768,
                 conv_filter_ratios=[0.5, 0.65, 0.85, 1],
                 conv1_kernel_size=20,
                 pointwise_kernal_size=8,
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
                 return_embeddings=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.conv_act = conv_act
        self.head_act = head_act
        self.num_targets = num_targets
        self.pool_size = pool_size
        self.conv_tower_layer = conv_tower_layer
        self.dilated_conv_tower_layer = dilated_conv_tower_layer
        self.conv_filters = conv_filters
        self.conv_filter_ratios = conv_filter_ratios
        self.conv1_kernel_size = conv1_kernel_size
        self.pointwise_kernal_size = pointwise_kernal_size
        self.conv_tower_kernal_size = conv_tower_kernal_size
        self.dilated_conv_tower_kernal_size = dilated_conv_tower_kernal_size
        self.dilated_dropout_rate = dilated_dropout_rate
        self.pointwise_dropout_rate = pointwise_dropout_rate
        self.padding = padding
        self.batchnorm = batchnorm
        self.conv_tower_pointwise = conv_tower_pointwise
        self.dilated_conv_tower_pointwise = dilated_conv_tower_pointwise 
        self.dilated_dropout = dilated_dropout
        self.pointwise_dropout = pointwise_dropout
        self.return_embeddings = return_embeddings
                 
                 
        self.body = Conv1DDilatedConv1D(conv_act=self.conv_act,
                                        pool_size=self.pool_size,
                                         conv_tower_layer=self.conv_tower_layer,
                                         dilated_conv_tower_layer=self.dilated_conv_tower_layer,
                                         conv_filters=self.conv_filters,
                                         conv_filter_ratios=self.conv_filter_ratios,
                                         conv1_kernel_size=self.conv1_kernel_size,
                                         conv_tower_kernal_size=self.conv_tower_kernal_size,
                                         dilated_conv_tower_kernal_size=self.dilated_conv_tower_kernal_size,
                                         dilated_dropout_rate=self.dilated_dropout_rate,
                                         pointwise_dropout_rate=self.pointwise_dropout_rate,
                                         padding=self.padding,
                                         batchnorm=self.batchnorm,
                                         conv_tower_pointwise=self.conv_tower_pointwise,
                                         dilated_conv_tower_pointwise=self.dilated_conv_tower_pointwise,
                                         dilated_dropout=self.dilated_dropout,
                                         pointwise_dropout=self.pointwise_dropout,
                                         name='dilated_body')
        
        self.head = PretrainHead(head_act=self.head_act,
                                 num_targets=self.num_targets,
                                 conv_kernel_size=self.pointwise_kernal_size,
                                 name='pretrain_head')
        
    def call(self, seqs, training=False):
        # seqs [B, L, 4]
        
        seq_embeddings = self.body(seqs, training=training)
        
        # head_preds [B, 16, T]
        head_preds = self.head(seq_embeddings, training=training)
        
        outputs = (head_preds, seq_embeddings) if self.return_embeddings else head_preds
        return outputs
    
    
    def train_step(self, data):
        # seqs [B, L, 4]
        # targets [B, L, T]
        seqs, targets = data
       
        with tf.GradientTape() as tape:
            if self.return_embeddings:
                head_preds, _ = self(seqs, training = True)
            else:
                head_preds = self(seqs, training = True)
                 
            loss = self.compiled_loss(targets, head_preds, regularization_losses=self.losses)
            
        
        # Compute gradients
        total_variables = self.trainable_variables
        gradients = tape.gradient(loss, total_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, total_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(targets, head_preds)
        
        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        
        # seqs [B, L, 4]
        # targets [B, L, T]
        seqs, targets = data
       
        if self.return_embeddings:
            head_preds, _ = self(seqs, training = False)
        else:
            head_preds = self(seqs, training = False)
            
        # Updates the metrics tracking the loss
        self.compiled_loss(targets, head_preds, regularization_losses=self.losses)
            
        # Update metrics
        self.compiled_metrics.update_state(targets, head_preds)
        
        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

class SeqFinetuneModel(tf.keras.Model):
    def __init__(self, 
                 pretrained,
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
                 imb_dropout_rate=0.5,
                 padding='same',
                 batchnorm=True,
                 conv_tower_pointwise=False,
                 dilated_conv_tower_pointwise=True,
                 dilated_dropout=True,
                 pointwise_dropout=True,
                 return_embeddings=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.conv_act = conv_act
        self.p_head_act = p_head_act
        self.i_head_act = i_head_act
        self.num_targets = num_targets
        self.conv1_num = conv1_num
        self.pool_size = pool_size
        self.conv_tower_layer = conv_tower_layer
        self.dilated_conv_tower_layer = dilated_conv_tower_layer
        self.conv_filters = conv_filters
        self.imb_filter = imb_filter
        self.conv_filter_ratios = conv_filter_ratios
        self.conv1_kernel_size = conv1_kernel_size
        self.pointwise_kernal_size = pointwise_kernal_size
        self.conv_tower_kernal_size = conv_tower_kernal_size
        self.dilated_conv_tower_kernal_size = dilated_conv_tower_kernal_size
        self.dilated_dropout_rate = dilated_dropout_rate
        self.pointwise_dropout_rate = pointwise_dropout_rate
        self.imb_dropout_rate = imb_dropout_rate
        self.padding = padding
        self.batchnorm = batchnorm
        self.conv_tower_pointwise = conv_tower_pointwise
        self.dilated_conv_tower_pointwise = dilated_conv_tower_pointwise 
        self.dilated_dropout = dilated_dropout
        self.pointwise_dropout = pointwise_dropout
        self.return_embeddings = return_embeddings
                 
                 
        self.body = Conv1DDilatedConv1D(conv_act=self.conv_act,
                                        pool_size=self.pool_size,
                                         conv_tower_layer=self.conv_tower_layer,
                                         dilated_conv_tower_layer=self.dilated_conv_tower_layer,
                                         conv_filters=self.conv_filters,
                                         conv_filter_ratios=self.conv_filter_ratios,
                                         conv1_kernel_size=self.conv1_kernel_size,
                                         conv_tower_kernal_size=self.conv_tower_kernal_size,
                                         dilated_conv_tower_kernal_size=self.dilated_conv_tower_kernal_size,
                                         dilated_dropout_rate=self.dilated_dropout_rate,
                                         pointwise_dropout_rate=self.pointwise_dropout_rate,
                                         padding=self.padding,
                                         batchnorm=self.batchnorm,
                                         conv_tower_pointwise=self.conv_tower_pointwise,
                                         dilated_conv_tower_pointwise=self.dilated_conv_tower_pointwise,
                                         dilated_dropout=self.dilated_dropout,
                                         pointwise_dropout=self.pointwise_dropout,
                                         name='dilated_body')
        dummy_out = self.body(dummy_train_x, training=False)
        self.body.set_weights(pretrained.layers[0].get_weights())
        self.body.trainable = False
        
    
        self.airehead = AireHead(conv_act=self.conv_act,
                                 i_head_act=self.i_head_act,
                                 conv1_num=self.conv1_num,
                                 dropout_rate=self.imb_dropout_rate,
                                 conv_kernel_size=self.pointwise_kernal_size,
                                 imb_filter=self.imb_filter,
                                 name='aire_head')
        
        
    def call(self, data, training=False):
        seq = data
        
        # seq_embeddings 
        seq_embeddings = self.body(seq, training=False)
        
        # type_preds [B, 1]
        gene_type_pred = self.airehead(seq_embeddings, training=training)
            
        outputs = (gene_type_pred, seq_embeddings) if self.return_embeddings else gene_type_pred
        return outputs
    
    
    def train_step(self, data):
        seq, gene_type = data
        
        gene_type_expand = tf.expand_dims(gene_type, axis=-1) #[B,1]
    
        with tf.GradientTape() as tape:
            if self.return_embeddings:
                gene_type_pred, seq_embeddings = self(seq, training = True)
            else:
                gene_type_pred = self(seq, training = True)
                 
            loss = self.compiled_loss(gene_type_expand, gene_type_pred, regularization_losses=self.losses)
            
        
        # Compute gradients
        total_variables = self.trainable_variables
        gradients = tape.gradient(loss, total_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, total_variables))
        
        # Update metrics (includes the metric that tracks the loss)
        # Only customized metrics can be used
        self.compiled_metrics.update_state(gene_type_expand, gene_type_pred)
        
        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # seq [B, 2048, 4]
        # gene_type [B,]
        seq, gene_type = data
        
        gene_type_expand = tf.expand_dims(gene_type, axis=-1) #[B,1]
        
        if self.return_embeddings:
            gene_type_pred, seq_embeddings = self(seq, training = False)
        else:
            gene_type_pred = self(seq, training = False)
        
        # Updates the metrics tracking the loss
        self.compiled_loss(gene_type_expand, gene_type_pred, regularization_losses=self.losses)
            
        # Update metrics (includes the metric that tracks the loss)
        # Only customized metrics can be used
        self.compiled_metrics.update_state(gene_type_expand, gene_type_pred)
        
        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
class AireFinetuneModel(tf.keras.Model):
    def __init__(self, 
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
                 imb_dropout_rate=0.5,
                 padding='same',
                 batchnorm=True,
                 conv_tower_pointwise=False,
                 dilated_conv_tower_pointwise=True,
                 dilated_dropout=True,
                 pointwise_dropout=True,
                 return_embeddings=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.conv_act = conv_act
        self.p_head_act = p_head_act
        self.i_head_act = i_head_act
        self.num_targets = num_targets
        self.conv1_num = conv1_num
        self.pool_size = pool_size
        self.conv_tower_layer = conv_tower_layer
        self.dilated_conv_tower_layer = dilated_conv_tower_layer
        self.conv_filters = conv_filters
        self.imb_filter = imb_filter
        self.conv_filter_ratios = conv_filter_ratios
        self.conv1_kernel_size = conv1_kernel_size
        self.pointwise_kernal_size = pointwise_kernal_size
        self.conv_tower_kernal_size = conv_tower_kernal_size
        self.dilated_conv_tower_kernal_size = dilated_conv_tower_kernal_size
        self.dilated_dropout_rate = dilated_dropout_rate
        self.pointwise_dropout_rate = pointwise_dropout_rate
        self.imb_dropout_rate = imb_dropout_rate
        self.padding = padding
        self.batchnorm = batchnorm
        self.conv_tower_pointwise = conv_tower_pointwise
        self.dilated_conv_tower_pointwise = dilated_conv_tower_pointwise 
        self.dilated_dropout = dilated_dropout
        self.pointwise_dropout = pointwise_dropout
        self.return_embeddings = return_embeddings
                 
                 
        self.body = Conv1DDilatedConv1D(conv_act=self.conv_act,
                                        pool_size=self.pool_size,
                                         conv_tower_layer=self.conv_tower_layer,
                                         dilated_conv_tower_layer=self.dilated_conv_tower_layer,
                                         conv_filters=self.conv_filters,
                                         conv_filter_ratios=self.conv_filter_ratios,
                                         conv1_kernel_size=self.conv1_kernel_size,
                                         conv_tower_kernal_size=self.conv_tower_kernal_size,
                                         dilated_conv_tower_kernal_size=self.dilated_conv_tower_kernal_size,
                                         dilated_dropout_rate=self.dilated_dropout_rate,
                                         pointwise_dropout_rate=self.pointwise_dropout_rate,
                                         padding=self.padding,
                                         batchnorm=self.batchnorm,
                                         conv_tower_pointwise=self.conv_tower_pointwise,
                                         dilated_conv_tower_pointwise=self.dilated_conv_tower_pointwise,
                                         dilated_dropout=self.dilated_dropout,
                                         pointwise_dropout=self.pointwise_dropout,
                                         name='dilated_body')
        self.body.trainable = False
        
    
        self.airehead = AireHead(conv_act=self.conv_act,
                                 i_head_act=self.i_head_act,
                                 conv1_num=self.conv1_num,
                                 dropout_rate=self.imb_dropout_rate,
                                 conv_kernel_size=self.pointwise_kernal_size,
                                 imb_filter=self.imb_filter,
                                 name='aire_head')
        
        
    def call(self, data, training=False):
        # seq [B, 2048, 4]
        # gene_type [B,]
        seq = data
        
        # seq_embeddings 
        seq_embeddings = self.body(seq, training=False)
        
        # type_preds [B, 1]
        gene_type_pred = self.airehead(seq_embeddings, training=training)
            
        outputs = (gene_type_pred, seq_embeddings) if self.return_embeddings else gene_type_pred
        return outputs
    
    
    def train_step(self, data):
        # seq [B, 2048, 4]
        # gene_type [B,]
        seq, gene_type = data
        
        gene_type_expand = tf.expand_dims(gene_type, axis=-1) #[B,1]
    
        with tf.GradientTape() as tape:
            if self.return_embeddings:
                gene_type_pred, seq_embeddings = self(seq, training = True)
            else:
                gene_type_pred = self(seq, training = True)
                 
            loss = self.compiled_loss(gene_type_expand, gene_type_pred, regularization_losses=self.losses)
            
        
        # Compute gradients
        total_variables = self.trainable_variables
        gradients = tape.gradient(loss, total_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, total_variables))
        
        # Update metrics (includes the metric that tracks the loss)
        # Only customized metrics can be used
        self.compiled_metrics.update_state(gene_type_expand, gene_type_pred)
        
        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # seq [B, 2048, 4]
        # gene_type [B,]
        seq, gene_type = data
        
        gene_type_expand = tf.expand_dims(gene_type, axis=-1) #[B,1]
        
        if self.return_embeddings:
            gene_type_pred, seq_embeddings = self(seq, training = False)
        else:
            gene_type_pred = self(seq, training = False)
        
        # Updates the metrics tracking the loss
        self.compiled_loss(gene_type_expand, gene_type_pred, regularization_losses=self.losses)
            
        # Update metrics (includes the metric that tracks the loss)
        # Only customized metrics can be used
        self.compiled_metrics.update_state(gene_type_expand, gene_type_pred)
        
        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """
  Applies a warmup schedule on a given learning rate decay schedule
  """

  def __init__(
    self,
    warmup_steps: int,
    decay_schedule: None,
    initial_learning_rate: float = 1e-4,
    power: float = 1.0,
    name: str = None,
  ):
    super().__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule = decay_schedule
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "WarmUp") as name:
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
      if callable(self.decay_schedule):
        warmed_learning_rate = self.decay_schedule(step - self.warmup_steps)
      else:
        warmed_learning_rate = self.decay_schedule
      return tf.cond(
        global_step_float < warmup_steps_float,
        lambda: warmup_learning_rate,
        lambda: warmed_learning_rate,
        name=name,
      )

  def get_config(self):
    return {
      "initial_learning_rate": self.initial_learning_rate,
      "decay_schedule": self.decay_schedule,
      "warmup_steps": self.warmup_steps,
      "power": self.power,
      "name": self.name,
    }

class EarlyStoppingMin(tf.keras.callbacks.EarlyStopping):
  """
  Stop training when a monitored quantity has stopped improving.
  """
  def __init__(self, min_epoch=0, **kwargs):
    super().__init__(**kwargs)
    self.min_epoch = min_epoch

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if epoch >= self.min_epoch and self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)


######################## Utility functions ########################

def get_activation(act_string):
    if act_string == 'gelu':
        return GELU()
    elif act_string == 'softplus':
        return Softplus()
    elif act_string == 'relu':
        return tf.keras.layers.ReLU()
    elif act_string == 'sigmoid':
        return tf.keras.layers.Activation('sigmoid')
    elif act_string == 'tanh':
        return tf.keras.layers.Activation('tanh')
    else:
         print('Unrecognized activation "%s"' % act_string, file=sys.stderr)
         exit(1)

def decode_loss(loss_id, total_weight=1):
    if loss_id == 1:
        return tf.keras.losses.Poisson()
    elif loss_id == 2:
        return MSEMultinomialPretrain(total_weight=total_weight)
    elif loss_id == 3:
        return PoissonMultinomialPretrain(total_weight=total_weight)
    else:
         print('Unrecognized loss id "%s"' % loss_id, file=sys.stderr)
         exit(1)

def create_byte_feature(values):
    values = values.flatten().tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def decode_fn_finetune(record_bytes):
    features = {"sequence": tf.io.FixedLenFeature([], dtype=tf.string),
                "target": tf.io.FixedLenFeature([], dtype=tf.string)
                }
    parsed_example = tf.io.parse_single_example(record_bytes, features)
            
    # decode
    sequence = tf.io.decode_raw(parsed_example['sequence'], tf.uint8)
    gene_type = tf.io.decode_raw(parsed_example['target'], tf.uint8)
    
    sequence = tf.reshape(sequence, [2048, 4]) #[2048, 4]
    sequence = tf.cast(sequence, tf.float32)
    gene_type = tf.reshape(gene_type, []) 
    gene_type = tf.cast(gene_type, tf.int32)
    return sequence,gene_type

def decode_fn_finetune_2(record_bytes):
    features = {"sequence": tf.io.FixedLenFeature([], dtype=tf.string),
                "target": tf.io.FixedLenFeature([], dtype=tf.string)
                }
    parsed_example = tf.io.parse_single_example(record_bytes, features)
            
    # decode
    sequence = tf.io.decode_raw(parsed_example['sequence'], tf.uint8)
    gene_type = tf.io.decode_raw(parsed_example['target'], tf.uint8)
    
    sequence = tf.reshape(sequence, [2048, 4]) #[2048, 4]
    sequence = tf.cast(sequence, tf.float32)
    gene_type = tf.reshape(gene_type, [1]) #[1]
    gene_type = tf.cast(gene_type, tf.int32)
    return sequence,gene_type

