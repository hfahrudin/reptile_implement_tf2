import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import timeit
from MAML import *
from SinusoidDs import copy_model

class MAMLEnhanced:
    def __init__(self, model, inner_loop=5, meta_iter=10, lr_outer=0.001, lr_inner=0.01, lr_mc=0.00001):
        self.model = model
        self.opt_outer = keras.optimizers.Adam(learning_rate=lr_outer)
        self.opt_inner = keras.optimizers.SGD(learning_rate=lr_inner)
        self.opt_mc = keras.optimizers.Adam(learning_rate=lr_mc)
        self.mc = self._initialize_mc(self.model)
        self.lslr = self._init_lslr(self.model, lr_inner)
        self.inner_loop = inner_loop
        self.meta_iter = meta_iter
        self.runtime = []

    def _initialize_mc(self, mdl):
        weights = mdl.get_weights()
        mc_in, mc_out, mc_f = [], [], []
        for w in weights:
            shape = w.shape
            i = tf.Variable(1.0, dtype=tf.float64)
            if len(shape) == 1:  # Bias
                mc_f.append(tf.Variable(i))
            elif len(shape) == 2:  # Fully connected layer
                n_in, n_out = shape
                mc_in.append(tf.Variable(tf.linalg.diag([i] * n_in)))
                mc_out.append(tf.Variable(tf.linalg.diag([i] * n_out)))
            else:  # Convolutional or higher-dimensional layers
                n_in, n_out = shape[-2], shape[-1]
                n_f = int(np.prod(shape) / (n_in * n_out))
                mc_in.append(tf.Variable(tf.linalg.diag([i] * n_in)))
                mc_out.append(tf.Variable(tf.linalg.diag([i] * n_out)))
                mc_f.append(tf.Variable(tf.linalg.diag([i] * n_f)))
        return [mc_in, mc_out, mc_f]

    def _init_lslr(self, model, init_value):
        return [tf.Variable(init_value, dtype="float64") for _ in model.trainable_variables]

    def _importance_weights(self, length, multi_steps, curr):
        loss_weights = np.ones(length) * (1.0 / length)
        decay_rate = 1.0 / length / multi_steps
        min_non_final_loss = 0.03 / length
        for i in range(length - 1):
            curr_val = max(loss_weights[i] - (curr * decay_rate), min_non_final_loss)
            loss_weights[i] = curr_val

        loss_weights[-1] = min(
            loss_weights[-1] + (curr * (length - 1) * decay_rate),
            1.0 - ((length - 1) * min_non_final_loss),
        )
        return loss_weights

    def _transform_gradients(self, gradients, all_mc):
        mc_in, mc_out, mc_f = all_mc
        ctin, ctout, ctf = 0, 0, 0
        for i, grad in enumerate(gradients):
            shape = grad.shape
            if len(shape) == 1:  # Bias
                gradients[i] = tf.multiply(mc_f[ctf], grad)
                ctf += 1
            elif len(shape) == 2:  # Fully connected
                temp = tf.matmul(mc_in[ctin], grad)
                gradients[i] = tf.matmul(temp, mc_out[ctout])
                ctin += 1
                ctout += 1
            else:  # Convolutional or higher-dimension
                n_in, n_out = shape[-2], shape[-1]
                n_f = int(np.prod(shape) / (n_in * n_out))
                temp = tf.matmul(tf.reshape(grad, [-1, n_out]), mc_out[ctout])
                temp = tf.reshape(tf.matmul(mc_f[ctf], tf.reshape(temp, [n_f, -1])), [n_f, n_in, n_out])
                temp = tf.matmul(mc_in[ctin], tf.reshape(tf.transpose(temp, [1, 0, 2]), [n_in, -1]))
                gradients[i] = tf.reshape(tf.transpose(tf.reshape(temp, [n_in, n_f, n_out]), [1, 0, 2]), shape)
                ctin += 1
                ctout += 1
                ctf += 1
        return gradients

    def meta_train(self, iter_ds, iter_train):
        for i in range(iter_train):
            i_weights = self._importance_weights(self.inner_loop, self.meta_iter, i)
            start_time = timeit.default_timer()
            print(f"Epoch {i + 1}/{iter_train}")
            for t in iter_ds:
                x, y = np_to_tensor(t.batch())
                with tf.GradientTape(persistent=True) as outer_tape:
                    outer_loss = tf.Variable(0, dtype="float64")
                    model_copy = copy_model(self.model, x)

                    for j in range(self.inner_loop):
                        with tf.GradientTape() as inner_tape:
                            inner_loss, logits = compute_loss(model_copy, x, y)
                        gradients = inner_tape.gradient(inner_loss, model_copy.trainable_variables)
                        gradients = self._transform_gradients(gradients, self.mc)

                        for layer_idx, var_idx in enumerate(range(0, len(gradients), 2)):
                            model_copy.layers[layer_idx].kernel.assign_sub(self.lslr[layer_idx] * gradients[var_idx])
                            model_copy.layers[layer_idx].bias.assign_sub(self.lslr[layer_idx] * gradients[var_idx + 1])

                        if j == self.inner_loop - 1:
                            outer_loss.assign_add(i_weights[j] * inner_loss)

                lslr_grad = outer_tape.gradient(outer_loss, self.lslr)
                self.opt_outer.apply_gradients(zip(lslr_grad, self.lslr))

                mc_grad = outer_tape.gradient(outer_loss, self.mc)
                for grad, mc in zip(mc_grad, self.mc):
                    self.opt_mc.apply_gradients(zip(grad, mc))

            self.runtime.append(timeit.default_timer() - start_time)
            print(f"Epoch runtime: {self.runtime[-1]:.2f}s")
