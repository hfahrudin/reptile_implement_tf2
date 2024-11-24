# Meta-Learning Algorithms Implementation

This repository contains implementations of popular meta-learning algorithms, designed to help machines quickly adapt to new tasks using only a small amount of data. The implemented algorithms include:

- **Reptile**: A simple and efficient gradient-based meta-learning method.
- **Model-Agnostic Meta-Learning (MAML)**: A widely-used meta-learning algorithm for fast adaptation.
- **MAML Enhanced**: A customized version of MAML, incorporating:
  - **Meta-Curvature (MC)**: Introduces learnable transformations of gradients, improving optimization in the meta-learning process.
  - **Layer-Specific Learning Rates (LSLR)**: Adjusts learning rates for each layer dynamically, enhancing training efficiency.

These implementations have been tested primarily on the **Sinusoid dataset**, a common benchmark for evaluating meta-learning methods.

---

## What is Meta-Learning?

Meta-learning often referred to as "learning to learn," is a subfield of machine learning that focuses on enabling models to adapt quickly to new tasks using limited data. Unlike traditional learning methods, which train a model to perform a single task meta-learning trains a model to develop a generalized strategy that works across a variety of tasks. This approach is particularly useful in scenarios where data is scarce or tasks are highly diverse.

---

## How It Works

### Reptile : On First-Order Meta-Learning Algorithms

[https://arxiv.org/abs/1803.02999](https://arxiv.org/abs/1803.02999)


> ... It also includes Reptile, a new algorithm that we introduce here, which works by repeatedly sampling
a task, training on it, and moving the initialization towards the trained weights on that task.
We expand on the results from Finn et al. showing that first-order meta-learning algorithms
perform well on some well-established benchmarks for few-shot classification, and we provide
theoretical analysis aimed at understanding why these algorithms work.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25025173/80688197-6774e480-8b06-11ea-9e95-728ed2647a83.PNG" alt="MAML Enhanced Workflow" width="400" />
</p>


### Model-Agnostic Meta-Learning (MAML)

[https://arxiv.org/abs/1703.03400](https://arxiv.org/abs/1703.03400)

> MAML is a meta-learning algorithm that aims to train models that can rapidly adapt to new tasks with few examples. The core idea is to optimize the initialization of the model so that, after a few gradient steps on a new task, the model's performance is maximized. By training on a variety of tasks and optimizing for fast adaptation, MAML enables models to generalize well to unseen tasks. This work has been widely adopted in the field of meta-learning due to its effectiveness in few-shot learning scenarios.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e58cc1b-1a46-4ad5-8362-017dfcad6624" alt="MAML Enhanced Workflow" width="400" />
</p>

### MAML Enhanced with Meta-Curvature and LSLR

Inspired by [How to Train Your MAML](https://arxiv.org/pdf/1810.09502), This Below is a high-level diagram illustrating the workflow of MAML Enhanced, including Meta-Curvature and LSLR:

<p align="center">
  <img src="https://github.com/user-attachments/assets/a302fcaa-9054-42ea-9611-338de264bef7" alt="MAML Enhanced Workflow" width="430" />
</p>
<p align="center">
  <i>Implementation of Layer-Specific Learning Rates in MAML Framework</i>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f3344862-5dba-4eb8-88ae-a99b7c42c6ca" alt="MAML Enhanced Workflow" width="430" />
</p>
<p align="center">
    <i>Implementation of Meta Curvature in MAML Framework</i>
</p>


- **Inner Loop**: Task-specific adaptation using gradient descent, where Meta-Curvature transforms the gradients.
- **Outer Loop**: Meta-updates using aggregated loss across tasks, with LSLR fine-tuning learning rates for each layer.

---

## Usage

### Getting Started

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Running Experiments

You can run experiments on the Sinusoid dataset using the provided python notebook.

### For Other Task

This script might or might not suitable with your use case so please modify each method script to cater your needs.

---
## Future Work

- Support for additional datasets and tasks.
- Benchmarking against state-of-the-art algorithms.

---

## References

1. Finn, Chelsea, et al. *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*. (2017)  
   [Read Paper](https://arxiv.org/pdf/1703.03400)

2. Nichol, Alex, and John Schulman. *Reptile: A Scalable Metalearning Algorithm*. (2018)  
   [Read Paper](https://arxiv.org/pdf/1803.02999)

3. Park, Eunbyung, et al. *Meta-Curvature*. (2019)  
   [Read Paper](https://arxiv.org/pdf/2002.09789)

4. Antoniou, Antreas, et al. *How To Train Your MAML*. (2019)  
   [Read Paper](https://arxiv.org/pdf/1810.09502)
