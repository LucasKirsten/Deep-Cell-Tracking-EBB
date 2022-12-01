import tensorflow as tf
import tensorflow.keras.backend as K

import configs as cfgs

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def _focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        
        FLp = -alpha*K.pow(1-y_pred, gamma)*K.log(y_pred)
        FLn = -(1-alpha)*K.pow(y_pred, gamma)*K.log(1-y_pred)
        
        loss = y_true*FLp + (1-y_true)*FLn
        
        # sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return cfgs.CLS_WEIGHT * loss
    
    return _focal_loss

def smoothl1():
    
    huber = tf.losses.huber_loss
    
    def _smoothl1(y_true, y_pred):
        
        # get only indices with objects
        indices = tf.where(y_true[...,2]>0)
        
        y_true = tf.gather_nd(y_true, indices)
        y_pred = tf.gather_nd(y_pred, indices)
        
        return cfgs.REG_WEIGHT * huber(y_true, y_pred)
    
    return _smoothl1
        
        
        
        
        
        
        
        
        
        
        
        
        