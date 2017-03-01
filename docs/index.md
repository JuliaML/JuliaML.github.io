
JuliaML is your one-stop-shop for learning models from data.  We provide general abstractions and algorithms for modeling and optimization, implementations of common models, tools for working with datasets, and much more.

The design and structure of the organization is geared towards a modular and composable approach to all things data science.  Plug and play models, losses, penalties, and algorithms however you see fit, and at whatever granularity is appropriate.  Beginner users and those looking for ready-made solutions can use the convenience package [Learn](https://github.com/JuliaML/Learn.jl).  For custom modeling solutions, choose methods and components from any package:

### [Learn](https://github.com/JuliaML/Learn.jl)

An all-in-one workbench, which simply imports and re-exports the packages below.  This is a convenience wrapper for an easy way to get started with the JuliaML ecosystem.

---

## Core

### [LearnBase](https://github.com/JuliaML/LearnBase.jl)

The abstractions and methods for JuliaML packages.  This is a core dependency of most packages.

### [LossFunctions](https://github.com/JuliaML/LossFunctions.jl)

Supervised and unsupervised loss functions for both distance-based (probabilities and regressions) and margin-based (SVM) approaches.

### [ObjectiveFunctions](https://github.com/JuliaML/ObjectiveFunctions.jl)

Generic definitions of objective functions using abstractions from LearnBase.

### [PenaltyFunctions](https://github.com/JuliaML/PenaltyFunctions.jl)

 Provides generic implementations for a diverse set of penalty functions that are commonly used for regularization purposes.

### [MLKernels](https://github.com/JuliaML/MLKernels.jl)

A Julia package for Mercer kernel functions (or the covariance functions used in Gaussian processes) that are used in the kernel methods of machine learning.

### [Transformations](https://github.com/JuliaML/Transformations.jl) (experimental)

Dynamic tensor computation framework. Static transforms, activation functions, neural nets, and more.

---

## Learning Algorithms

### [LearningStrategies](https://github.com/JuliaML/LearningStrategies.jl)

A generic and modular framework for building custom iterative algorithms in Julia.

### [StochasticOptimization](https://github.com/JuliaML/StochasticOptimization.jl)

Extension of LearningStrategies implementing stochastic gradient descent and online optimization algorithms and components.  Parameter update models (Adagrad, ADAM, etc).  Minibatch gradient averaging.

### [ContinuousOptimization](https://github.com/JuliaML/ContinuousOptimization.jl) (WIP, help needed)

Unconstrained Continuous Full-Batch Optimization Algorithms based on the LearningStrategies framework.  This is a prototype package meant to explore how we could move [Optim](https://github.com/JuliaNLSolvers/Optim.jl) algorithms to a more modular and maintainable framework.

---

## Reinforcement Learning

### [Reinforce](https://github.com/JuliaML/Reinforce.jl)

Abstractions, algorithms, and utilities for reinforcement learning in Julia

### [OpenAIGym](https://github.com/JuliaML/OpenAIGym.jl)

OpenAI's Gym wrapped as a Reinforce.jl environment

### [AtariAlgos](https://github.com/JuliaML/AtariAlgos.jl)

Arcade Learning Environment (ALE) wrapped as a Reinforce.jl environment

---

## Tools

### [MLDataUtils](https://github.com/JuliaML/MLDataUtils.jl)

Dataset iteration and splitting (test/train, K-folds cross validation, batching, etc).

### [MLLabelUtils](https://github.com/JuliaML/MLLabelUtils.jl)

Utility package for working with classification targets and label-encodings ([Docs](http://mllabelutilsjl.readthedocs.io/))

### [MLDatasets](https://github.com/JuliaML/MLDatasets.jl)

Machine Learning Datasets for Julia

### [MLMetrics](https://github.com/JuliaML/MLMetrics.jl)

Metrics for scoring machine learning models in Julia.  MSE, accuracy, and more.

### [ValueHistories](https://github.com/JuliaML/ValueHistories.jl)

Utilities to efficiently track learning curves or other optimization information

### [MLPlots](https://github.com/JuliaML/MLPlots.jl)

Plotting recipes to be used with [Plots](https://github.com/tbreloff/Plots.jl).  Also check out [PlotRecipes](https://github.com/JuliaPlots/PlotRecipes.jl).

---

## Other notable packages

### [MXNet](https://github.com/dmlc/MXNet.jl)

MXNet Julia Package - flexible and efficient deep learning in Julia

### [TensorFlow](https://github.com/malmaud/TensorFlow.jl)

A Julia wrapper for TensorFlow

### [Knet](https://github.com/denizyuret/Knet.jl)

Koç University deep learning framework.  It supports GPU operation and automatic differentiation using dynamic computational graphs for models defined in plain Julia.

### [Flux](https://github.com/MikeInnes/Flux.jl)

A high level API for machine learning, implemented in Julia.  Flux aims to provide a concise and expressive syntax for architectures that are hard to express within other frameworks.  The current focus is on ANNs with TensorFlow or MXNet as a backend.

### [Mocha](https://github.com/pluskid/Mocha.jl)

Deep Learning framework for Julia (author recommends MXNet instead)

### [KSVM](https://github.com/Evizero/KSVM.jl)

Support Vector Machines (SVM) in pure Julia

### [Other AI packages](https://github.com/svaksha/Julia.jl/blob/master/AI.md)
