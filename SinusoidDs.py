import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt

class SineModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = keras.layers.Dense(40, input_shape=(1,))
        self.hidden2 = keras.layers.Dense(40)
        self.out = keras.layers.Dense(1)
        
    def forward(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = self.out(x)
        return x

class SinusoidGenerator() :

    def __init__ (self, K=None, amplitude = None, phase = None):
        self.K = K
        self.amplitude = amplitude if amplitude else np.random.uniform(0.1, 1.0)
        self.phase = phase*np.pi if phase else np.random.uniform(0,  0.2*np.pi)
        self.x = self._sample_x()
        
    def _sample_x(self):
        return np.random.uniform(-5,5, self.K)
    
    def f(self, x):
        return self.amplitude * np.sin(x - self.phase)
    
    def batch(self, x=None, force_new = False):
        if x is None:
            if force_new:
                x = self._sample_x()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]
    
    def equally_spaced_samples(self, K= None):
        if K is None:
            K = self.K
        return self.batch (x = np.linspace(-5, 5, K))
    
    def random_spaced_samples(self, K= None):
        if K is None:
            K = self.K
        return self.batch(x = np.random.uniform(-5, 5, K))
    
def copy_model(model, x=None):
    '''Copy model weights to a new model.
    
    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    
    copied_model = SineModel()
    
    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model.forward(tf.convert_to_tensor(x))
    
    copied_model.set_weights(model.get_weights())
    return copied_model



def test_session_sinusoid(model, train_func, x_copy):
    all_losses = []
    p = [0.3 , 0.5, 0.7, 0.9, 1]
    for i in range (0,5):
        title = "Amplitude : "+ str((i+2)) + " Phase : "+ str(p[i])+"*phi)"
        print(title)
        last_losses = 0
        all_res = []
        for j in range(0, 5):
            print("======= ", j+1, " run ========")
            test_task = [SinusoidGenerator(K=100, amplitude = i+2 , phase=p[i])]
            cp_model = copy_model(model, x_copy)
            res = train_func(epochs=100, mdl=cp_model, dataset=test_task, log_steps=None)
            all_res.append(res[1])
            last_losses+=res[1][-1] 
        plt.plot(all_res[0])
        plt.plot(all_res[1])
        plt.plot(all_res[2])
        plt.plot(all_res[3])
        plt.plot(all_res[4])
        plt.title(title)
        plt.legend(["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"], loc=(1.05, 0.5))
        plt.xlabel("Adaptation steps")
        plt.ylabel("Loss")
        plt.show()
        all_losses.append(last_losses/5)
    
    category = ["Test_Ds1", "Test_Ds2", "Test_Ds3","Test_Ds4","Test_Ds5"]
    fig, ax = plt.subplots()
    ax.plot(category, all_losses, label="loss")
    ax.legend(loc=(1.05, 0.5))
    plt.show()
    return all_losses
    