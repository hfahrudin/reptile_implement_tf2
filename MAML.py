import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from math import cos
from math import pi
from math import floor

# Other dependencies
import random
import sys
import timeit

import numpy as np
import matplotlib.pyplot as plt
import os

loss_function = tf.losses.MeanSquaredError()


def normalize(inp, gamma, beta):
    mean, variance = tf.nn.moments(inp,  [0])
    norm = tf.nn.batch_normalization(inp, mean, variance, beta, gamma, 0.001)
    return tf.nn.relu(norm)

def forward_func(inp, weights, reuse=False):
    hidden = tf.matmul(inp, weights[0]) + weights[1]
    hidden = tf.nn.relu(hidden)
    hidden = tf.matmul(hidden, weights[2]) + weights[3]
    hidden = tf.nn.relu(hidden)

    return tf.keras.activations.linear(tf.matmul(hidden, weights[4]) + weights[5])

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj, dtype="float32") for obj in list_of_numpy_objs)

def tensor_to_np(tensor):
    return (np.array(obj) for obj in tensor)


def model_func(model, x_train, t_train): #compute loss
    y_pred = model(x_train)
    loss = tf.losses.MeanAbsoluteError()(t_train, y_pred)
    return y_pred, loss
def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model(x)
    mse = loss_fn(y, logits)
    return mse, logits
#@tf.function
def train_batch(x, y, model, optimizer):
    tensor_x, tensor_y = (x,y)
    with tf.GradientTape() as tape:
        _, loss= model_func(model, tensor_x, tensor_y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, model


class MAMLmodel():
    def __init__(self, model, inner_update=5, meta_batch_size=16):
        self.maml_model = model
        self.inner_update = inner_update
        self.meta_batch_size = meta_batch_size
        self.opt_outer= keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def task_metalearn(self, inp):
        xs, ys,xt,yt = inp
    
        
        outa = self.maml_model(xs, training=True)
        lossa = loss_function(ys, outa)
        grads = tf.gradients(lossa, self.maml_model.trainable_variables)
        
        fast_weights = [p - 1e-3*grads[idx] for idx, p in enumerate(self.maml_model.trainable_variables)]

        for _ in range(self.inner_update - 1):
            outa = forward_func(xs, fast_weights)
            lossa = loss_function(ys, outa)
            grads = tf.gradients(lossa, fast_weights)
            fast_weights = [p - 1e-3*grads[idx] for idx, p in enumerate(fast_weights)]
    

        #sec order    
        outb = forward_func(xs, fast_weights)
        lossb = loss_function(ys, outb)
        
        return outa, outb, lossa, lossb
        
    @tf.function
    def meta_update(self,inp):
        _,_,_,outer_loss = tf.map_fn(self.task_metalearn, elems=inp, dtype=(tf.float32, tf.float32,tf.float32,tf.float32), parallel_iterations=self.meta_batch_size)
        lossb = tf.reduce_sum(outer_loss)/self.meta_batch_size
        grads = tf.gradients(lossb, self.maml_model.trainable_variables)
        self.opt_outer.apply_gradients(zip(grads, self.maml_model.trainable_variables))
        return lossb
    
    def meta_training(self, ds_iter):
        title_temp = 'Update {} : Outer loss: {}   time: {}'
        update = 1
        meta_iter = 5000
        while update <= meta_iter:
            for inp in ds_iter:
                start = timeit.default_timer()
                loss = self.meta_update(inp)
                stop = timeit.default_timer()
                dur = stop - start
                print(title_temp.format(update, loss, dur))
                update+=1
                if update  >= meta_iter:
                    break

    @tf.function
    def task_eval(self, xs, ys,x_test, y_test):
        
        fit_res = []
        inner_update = 10
        
        outb = self.maml_model(x_test, training=True)
        lossb = loss_function(y_test, outb)
        fit_res.append((
            0, outb, lossb
        ))
        
        outa = self.maml_model(xs, training=True)
        lossa = loss_function(ys, outa)
        grads = tf.gradients(lossa, self.maml_model.trainable_variables)
        fast_weights = [p - 1e-3*grads[idx] for idx, p in enumerate(self.maml_model.trainable_variables)]

        
        outb = forward_func(x_test, fast_weights)
        lossb = loss_function(y_test, outb)
        fit_res.append((
            1, outb, lossb
        ))
        
        for j in range(inner_update - 1):
            outa = forward_func(xs, fast_weights)
            lossa = loss_function(ys, outa)
            grads = tf.gradients(lossa, fast_weights)
            fast_weights = [p - 1e-3*grads[idx] for idx, p in enumerate(fast_weights)]
    

        #sec order    
        outb = forward_func(x_test, fast_weights)
        lossb = loss_function(y_test, outb)
        fit_res.append((
            10, outb, lossb
        ))
        return fit_res
    
