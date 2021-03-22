# BML-project
Stochastic Gradient Langevin Dynamics



Nowadays, most of the problems, that Machine Learning has to solve, are based principally on very large scale datasets. Therefore, more and more advances are required in order to meet such needs.

In this context, the paper titled 'Bayesian Learning via Stochastic Gradient LangevinDynamics' came along with a new method for learning from this kind of datasets based on iterative learning from small mini-batches, by combining the usual stochastic gradient optimization algorithm and Bayesian posterior sampling.
Hence, the objective of this github project is to implement the method, then analyze it and apply it to a real data so as to have a concrete results of the efficiency and the performance of the method.

We first implement the algorithm by ourselves using python and then apply it on three different bayesian learnin problems:  the 'Logistic regression' with the a9a dataset,  Then, using a 'Gaussian Mixture' dataset. And finally, the 'Logistic regression' with the Iris dataset.

There are three notebooks for each experiments.

The file 'The skeleton of Stochastic Gradient Langevin Dynamics.py' contains the global architecture of this method, a new user needs just to adapt it to the corresponding problem, i.e mainly changing the gradients.
