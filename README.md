## Implementation of Reptile in Tensorflow 2.0

This implementation are influenced by :
- Pytorch implementation by Adrien Lucas Effot: [Paper repro: Deep Metalearning using “MAML” and “Reptile”](https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0)
- MAML implementation on Tensorflow 2.0 by Marianne Monteiro : [Reproduction of MAML using TensorFlow 2.0.](https://github.com/mari-linhares/tensorflow-maml)

Github : https://github.com/hfahrudin

Facebook : https://www.facebook.com/hasby.fahrudin

### Reptile : On First-Order Meta-Learning Algorithms

https://arxiv.org/abs/1803.02999

> ... It also includes Reptile, a new algorithm that we introduce here, which works by repeatedly sampling
a task, training on it, and moving the initialization towards the trained weights on that task.
We expand on the results from Finn et al. showing that first-order meta-learning algorithms
perform well on some well-established benchmarks for few-shot classification, and we provide
theoretical analysis aimed at understanding why these algorithms work.
