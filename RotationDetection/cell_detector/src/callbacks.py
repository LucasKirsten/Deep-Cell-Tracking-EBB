import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

import configs as cfgs

class LearningRateScheduler(tf.keras.callbacks.Callback):
    
    def __init__(self, patience=0):
        super(LearningRateScheduler, self).__init__()
        self.global_step = 0
        
    def _warmup_lr(self, init_lr, global_step, warmup_step, num_gpu):
        def warmup(end_lr, global_step, warmup_step):
            start_lr = end_lr * 0.1
            global_step = tf.cast(global_step, tf.float32)
            return start_lr + (end_lr - start_lr) * global_step / warmup_step

        def decay(start_lr, global_step, num_gpu):
            if global_step < np.int32(cfgs.DECAY_STEP[0] // num_gpu):
                return start_lr
            elif np.int32(cfgs.DECAY_STEP[0] // num_gpu) <= global_step < np.int32(cfgs.DECAY_STEP[1] // num_gpu):
                return start_lr / 10.
            elif np.int32(cfgs.DECAY_STEP[1] // num_gpu) <= global_step < np.int32(cfgs.DECAY_STEP[2] // num_gpu):
                return start_lr / 100.
            else:
                return start_lr / 1000.

        return tf.cond(tf.less_equal(global_step, warmup_step),
                       true_fn=lambda: warmup(init_lr, global_step, warmup_step),
                       false_fn=lambda: decay(init_lr, global_step, num_gpu))
    
    def on_train_batch_begin(self, batch, logs=None):
        self.global_step += cfgs.BATCH_SIZE
        lr = self._warmup_lr(cfgs.LR, self.global_step, cfgs.WARM_SETP, cfgs.NUM_GPUS)
        K.set_value(self.model.optimizer.learning_rate, lr)