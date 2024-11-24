import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import np_to_tensor, compute_loss, train_batch

dtype="float32"

def generate_sinusoid(K):
    ranges = [-5, 5]
    amplitude = np.random.uniform(0.1, 5.0)
    phase = np.random.uniform(0, np.pi)
    xs = np.random.uniform(ranges[0], ranges[1], K)
    ys = amplitude * np.sin(xs - phase)
    xt = np.random.uniform(ranges[0], ranges[1], K)
    yt = amplitude * np.sin(xt - phase)
    return tf.convert_to_tensor(xs[:, None], dtype=dtype), tf.convert_to_tensor(ys[:, None], dtype=dtype), tf.convert_to_tensor(xt[:, None], dtype=dtype), tf.convert_to_tensor(yt[:, None], dtype=dtype)

def split_ds(ds):
    xs = ds[0]
    ys = ds[1]
    xt = ds[2]
    yt = ds[3]
    return xs, ys, xt, yt

def meta_sinusoid_ds(K, batch_size, train=True):
    if train:
        size = 10000
    else:
        size = 600
    all_ds = []
    for _ in range(size):
        xs, ys, xt, yt = generate_sinusoid(K)
        all_ds.append([xs,ys,xt,yt])
    ds = tf.data.Dataset.from_tensor_slices(all_ds)
    ds = ds.map(split_ds)
    ds = ds.batch(batch_size)
    return ds
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

class SinusoidGenerator:

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


def eval_sine_test(model, optimizer, x, y, x_test, y_test, num_steps=(0, 1, 10)):
    '''Evaluate how the model fits to the curve training for `fits` steps.
    
    Args:
        model: Model evaluated.
        optimizer: Optimizer to be for training.
        x: Data used for training.
        y: Targets used for training.
        x_test: Data used for evaluation.
        y_test: Targets used for evaluation.
        num_steps: Number of steps to log.
    '''
    fit_res = []
    
    tensor_x_test, tensor_y_test = np_to_tensor((x_test, y_test))
    
    # If 0 in fits we log the loss before any training
    if 0 in num_steps:
        loss, logits = compute_loss(model, tensor_x_test, tensor_y_test)
        fit_res.append((0, logits, loss))
        
    for step in range(1, np.max(num_steps) + 1):
        train_batch(x, y, model, optimizer)
        loss, logits = compute_loss(model, tensor_x_test, tensor_y_test)
        if step in num_steps:
            fit_res.append(
                (
                    step, 
                    logits,
                    loss
                )
            )
    return fit_res


def eval_sinewave_for_test(title, eval_function, sinusoid_generator=None, num_steps=(0, 1, 10), lr=0.01, plot=True):
    '''Evaluates how the sinewave addapts at dataset.
    
    The idea is to use the pretrained model as a weight initializer and
    try to fit the model on this new dataset.
    
    Args:
        model: Already trained model.
        sinusoid_generator: A sinusoidGenerator instance.
        num_steps: Number of training steps to be logged.
        lr: Learning rate used for training on the test data.
        plot: If plot is True than it plots how the curves are fitted along
            `num_steps`.
    
    Returns:
        The fit results. A list containing the loss, logits and step. For
        every step at `num_steps`.
    '''
    
    if sinusoid_generator is None:
        sinusoid_generator = SinusoidGenerator(K=10)
        
    # generate equally spaced samples for ploting
    x_test, y_test = sinusoid_generator.equally_spaced_samples(100)
    
    # batch used for training
    x, y = sinusoid_generator.batch()
    

    # run training and log fit results
    fit_res = eval_function(x, y, x_test, y_test)
    # plot
    train, = plt.plot(x, y, '^')
    ground_truth, = plt.plot(x_test, y_test)
    plots = [train, ground_truth]
    legend = ['Training Points', 'True Function']
    for n, res, loss in fit_res:
        cur, = plt.plot(x_test, res, '--')
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.ylim(-5, 5)
    plt.xlim(-6, 6)
    plt.title(title)
    if plot:
        plt.show()
    
    return fit_res