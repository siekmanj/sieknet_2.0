# sieknet

## A dependency-free recurrent neural network library written in C
This is a recurrent neural network and deep learning library written in C which implements various machine learning algorithms. I have mostly focused on recurrent and memory-based networks while writing this, because these interest me the most.

This project has no mandatory dependencies and is written completely from scratch - all you need to compile and run this code is `gcc` or any C compiler.

##### Contents  
- [But why?](#purpose)  

- [Features](#features)  

- [Future Plans](#future)  

<a name="purpose"/>

## But why?

This project began in June of 2018, when I decided to teach myself how neural networks work while working a summer internship. I decided to implement the algorithms I was learning about in the language I was most comfortable in - C. At some point that summer, I stumbled across Andrej Karpathy's [inspirational article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) on RNNs, and was struck by how powerful these relatively simple algorithms were at capturing intricate relationships across time, and began trying to understand how to implement them from scratch. This project has been one giant learning experience as I have slowly built up my knowledge of the underlying math and statistics involved in deep learning. It is not an attempt to create a full-featured library, though I have tried to make it as useful as possible. It was my goal to create an easy-to-use, fast, efficient and clean implementation of the algorithms involved in training and using recurrent neural networks.

As of September 2019, I have discovered that my matrix multiplication code is extremely slow compared to optimized implementations (such as in pytorch), and it is unlikely I will have the time or smarts to fix the discrepancy. If you choose to use this project, be aware that it may actually be slower than running a python-based library.

If you would like to use my library, you can find instructions below.

<a name="features"/>

## Features

Features include:

### Implicit Recurrence

Implicit recurrence is what I consider the core feature of this library. While neural networks are generally represented as DAGs in memory, in sieknet they can be represented as a graph with cycles (where cycles represent recurrence). This allows for recurrent connections not only between a layer and itself, but for recurrent connections between multiple layers, which can lead to intricate recurrent architectures.

<a name="future"/>

## Future

Plans for the near future include:
- [ ] policy gradient algorithms
- [ ] neural turing machine
- [ ] differentiable neural computer
- [ ] gated recurrent unit (GRU)

<a name="gpu">

## GPU Acceleration

Coming soon!

