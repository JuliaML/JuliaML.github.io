
JuliaML is your one-stop-shop for learning models from data.  We provide general abstractions and algorithms for modeling and optimization, along with implementations of common models:

- Flexible objective functions, and cost components:
	- Losses (supervised distance and margin losses, etc)
	- Penalties (L1/L2 regularization, etc)
	- (TODO) and other Costs
- Generic transformations:
	- Static transforms (log, exp, sqrt, etc)
	- Activation functions (sigmoid, tanh, ReLU, etc)
	- Centering/scaling/normalization
	- Dimensionality reduction
	- Affine transformations (y = wx + b)
	- (TODO) Directed Acyclic Graphs (DAG) of sub-transformations
- Learning algorithms
	- Online and Offline optimization
	- Gradient-based update models (SGD w/ momentum, Adagrad, ADAM, etc)
	- (TODO) Learning rate strategies (decaying, adaptive, etc)
	- (TODO) Hyperparameter fitting
	- (TODO) Ensembles
- Common/standard approaches
	- Empirical Risk Minimization
	- Ridge Regression, LASSO
	- (TODO) Support Vector Machines (SVM)
	- (TODO) Neural Nets (ANN), Deep Learning (DL)
	- (TODO) Decision Trees and Random Forests


The design and structure of the organization is geared towards a modular and composable approach to all things data science.  Plug and play models, losses, penalties, and algorithms however you see fit, and at whatever granularity is appropriate.  Beginner users and those looking for ready-made solutions can use the convenience package [Learn](https://github.com/JuliaML/Learn.jl).  For custom modeling solutions, choose methods and components from any package:

## [Learn](https://github.com/JuliaML/Learn.jl)

An all-in-one workbench, which simply imports and re-exports the packages below.  This is a convenience wrapper for an easy way to get started with the JuliaML ecosystem.

## [LearnBase](https://github.com/JuliaML/LearnBase.jl)

The abstractions and methods for JuliaML packages.  This is a core dependency of most packages.

## [Losses](https://github.com/JuliaML/Losses.jl)

Supervised and unsupervised loss functions for both distance-based (probabilities and regressions) and margin-based (SVM) approaches.

## [Transformations](https://github.com/JuliaML/Transformations.jl)

Static transforms, activation functions, and more.

## [ObjectiveFunctions](https://github.com/JuliaML/ObjectiveFunctions.jl)

Generic definitions of objective functions using abstractions from LearnBase.

## [StochasticOptimization](https://github.com/JuliaML/StochasticOptimization.jl)

Gradient descent and online optimization algorithms and components.  Parameter update models (Adagrad, ADAM, etc).  Proximal methods.

## [MLDataUtils](https://github.com/JuliaML/MLDataUtils.jl)

Dataset iteration and splitting (test/train, K-folds cross validation, batching, etc).

## [MLRisk](https://github.com/JuliaML/MLRisk.jl)

Empirical risk minimization.

## [MLMetrics](https://github.com/JuliaML/MLMetrics.jl)

Metrics for scoring machine learning models in Julia.  MSE, accuracy, and more.

## [MLPlots](https://github.com/JuliaML/MLPlots.jl)

Plotting recipes to be used with [Plots](https://github.com/tbreloff/Plots.jl).  Also check out [PlotRecipes](https://github.com/JuliaPlots/PlotRecipes.jl).

## [MLKernels](https://github.com/JuliaML/MLKernels.jl)

A Julia package for Mercer kernel functions (or the covariance functions used in Gaussian processes) that are used in the kernel methods of machine learning.






