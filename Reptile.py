
import tensorflow.keras as keras
from utils import SineModel, np_to_tensor, compute_loss, train_batch, copy_model
import matplotlib.pyplot as plt
 

import random
import time

import numpy as np

def train_reptile( epochs, dataset, mdl=None, lr_inner=0.001, lr_outer=0.01, batch_size=1, log_steps=100, k=1):
    #Step 1 : initialize model
    if mdl is not None :
        model = mdl
    else:
        model = SineModel()
    inner_optimizer = keras.optimizers.SGD(learning_rate=lr_inner)
    outer_optimizer = keras.optimizers.Adam(learning_rate=lr_outer)
    losses = []
    
    # Step 2 : iteration
    for epoch in range(epochs):
        if log_steps is not None:
            print("====== Epoch : " +str(epoch)+ " ====== ")
        total_loss = 0
        
        start = time.time()
        #Step 3 & 4 : get sample task from dataset
        for i, t in enumerate(random.sample(dataset, len(dataset))):
            x, y = np_to_tensor(t.batch())
            model.forward(x) 
            
            # save current parameter
            old_weights = model.get_weights()
                
            model_copy = copy_model(model, x)
            # Step 5 : Compute W with SGD
            for _ in range(k):
                loss = train_batch(x, y, model_copy, inner_optimizer)
            
            # Step 6 : update model parameter
            after_weights = model_copy.get_weights()
            step_size = lr_inner * (1 - epoch / epochs) # linear scheduling method
            new_weights = [ old_weights[i] + ((old_weights[i] - after_weights[i]) * step_size)
                           for i in range(len(model.weights))]
            model.set_weights(new_weights)
        
            # additional step for outer optimization
            if (i+1) % batch_size == 0:
                test_loss = train_batch(x, y, model, outer_optimizer)
            else:
                test_loss, _ = compute_loss(model, x, y)
            
            # Logs
            total_loss += test_loss
            loss = total_loss / (i+1.0)
            
            if log_steps is not None:
                if i % log_steps == 0 and i > 0:
                    print('Step {}: loss = {}, Time to run {} steps = {}'.format(i, loss, log_steps, time.time() - start))
                    start = time.time()
                    
        losses.append(loss)
    plt.plot(losses)
    plt.xlabel("Adaptation steps")
    plt.title("Mean Absolute Error Performance (REPTILE)")
    plt.ylabel("Loss")
    plt.show()
    return model, np.array(losses)