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

Meta-learning, often referred to as "learning to learn," is a subfield of machine learning that focuses on enabling models to adapt quickly to new tasks using limited data. Unlike traditional learning methods, which train a model to perform a single task, meta-learning trains a model to develop a generalized strategy that works across a variety of tasks. This approach is particularly useful in scenarios where data is scarce or tasks are highly diverse.

---

## How It Works

### MAML Enhanced with Meta-Curvature and LSLR

Below is a high-level diagram illustrating the workflow of MAML Enhanced, including Meta-Curvature and LSLR:

![MAML Enhanced Workflow](path/to/diagram.png)  
*Replace `path/to/diagram.png` with the actual path to your diagram.*

- **Inner Loop**: Task-specific adaptation using gradient descent, where Meta-Curvature transforms the gradients.
- **Outer Loop**: Meta-updates using aggregated loss across tasks, with LSLR fine-tuning learning rates for each layer.

---

## Getting Started

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

You can run experiments on the Sinusoid dataset using the provided scripts. Examples include training and testing different algorithms:


### Visualizing Performance

To better understand the effects of Meta-Curvature and LSLR, use the included visualization tools:

```bash
python examples/plot_results.py
```

### Customizing Configurations

Modify the parameters (e.g., inner-loop iterations, learning rates) directly in the respective scripts or modules to customize experiments. The `MAMLEnhanced` class provides additional functionality for fine-tuning the meta-learning process.

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
