# Purpose of this library :

Conventionally, Differentially Private ML training relies on Gradient Clipping to guarantee verifiable privacy guarantees.
By using 1-Lipschitz networks developped by the deel-lip project. We can propose a new alternative to gradient clipping based
DP ML. Indeed, by theoretically bounding the value of the sensitivity of our 1-Lipschitz layers, we can directly calibrate a
batchwise noising of the gradients to guarantee (epsilon,delta)-DP.

Therefore, the computation time is heavily reduced and the results on the MNIST and CIFAR10 datasets are the following :


# Status of the repository : 

- ci tests to develop.
- sensitivity.py to debug.
- requirements.txt tested on my machine, still to check by someone else.