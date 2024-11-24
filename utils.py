import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from SinusoidDs import SineModel

def plot (data, *args, **kwargs):
    x,y = data
    return plt.plot(x, y, *args, **kwargs)


def loss_function(pred_y, y):
    return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)
    

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits
   
def train_batch(x, y, model, optimizer):
    tensor_x, tensor_y = np_to_tensor((x, y))
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_reg(epochs, dataset,mdl=None, lr=0.001, log_steps=100):
    if mdl is not None :
        model = mdl
    else:
        model = SineModel()
    optimizer = keras.optimizers.Adam(learning_rate = lr)
    losses = []
    for epoch in range(epochs):
        if log_steps is not None:
            print("====== Epoch : " +str(epoch)+ " ====== ")
      
        total_loss = 0
        curr_loss = 0
        tmp = 0
        for i, sinusoid_generator in enumerate(dataset):
            x, y = sinusoid_generator.batch()
            loss = train_batch(x, y, model, optimizer)
            total_loss += loss
            curr_loss = total_loss / (i + 1.0)
            
            tmp = i
            if log_steps is not None:
                if i % log_steps == 0 and i > 0:
                    print('Step {}: loss = {}'.format(i, curr_loss))
        losses.append(curr_loss)  
    plt.plot(losses)
    plt.xlabel("Adaptation steps")
    plt.title("Mean Absolute Error Performance (Normal)")
    plt.ylabel("Loss")
    plt.show()
    return model, np.array(losses)