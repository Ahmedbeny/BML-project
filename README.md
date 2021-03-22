# BML-project
Stochastic Gradient Langevin Dynamics



Nowadays, most of the problems, that Machine Learning has to solve, are based principally on very large scale datasets. Therefore, more and more advances are required in order to meet such needs.

In this context, the paper titled 'Bayesian Learning via Stochastic Gradient LangevinDynamics'[1] came along with a new method for learning from this kind of datasets based on iterative learning from small mini-batches, by combining the usual stochastic gradient optimization algorithm and Bayesian posterior sampling.
Hence, the objective of this github project is to implement the method, then analyze it and apply it to a real data so as to have a concrete results of the efficiency and the performance of the method.

We first implement the algorithm by ourselves using python and then apply it on three different bayesian learnin problems:  the 'Logistic regression' with the a9a dataset (1) ,  then, using a 'Gaussian Mixture' dataset(2), and finally, the 'Logistic regression' with the Iris dataset(3).

There are three notebooks for each experiment:

1)a9a logistic regression .ipynb

2)Mixture of Gaussian.ipynb

3)Iris dataset.ipynb

The file 'The skeleton of Stochastic Gradient Langevin Dynamics.py' contains the global architecture of this method, a new user needs just to adapt it to the corresponding problem, i.e mainly changing the gradients.

Files a9a and a9a.t contain the a9a dataset (training and testing respectively) derived by ( Lin et al., 2008) from the UCI adult dataset. ( link: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a)


[1]: M. Welling and Y. W. Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11),
pages 681–688, 2011.(link: http://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)
