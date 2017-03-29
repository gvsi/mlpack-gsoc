**GSoC 2017** Project Proposal

Giovanni Alcantara

___
# Generative Adversarial Networks for mlpack

## Personal Background
My name is Giovanni, but you can call me Gio.

I am a Computer Science student from Italy, studying for an undergraduate degree at the University of Edinburgh in Scotland, and currently on an exchange year abroad at the University of Texas at Austin.

I am on track to graduate with a first class degree in May 2018, after which I will be pursuing a graduate degree in Computer Science with a focus in Artificial Intelligence and Machine Learning.

What captivates me in the dynamic, ever-changing field of Computer Science is how it symbolizes the heart and brain of the fundamental technologies of today’s connected world, as well as representing the enabling force and catalyst of multiple scientific disciplines.

## Proposed project
The proposed project has three main goals:
1. **Framework**: To build a general framework in mlpack for Generative Adversarial Network, leveraging the existing Artificial Neural Network classes. The framework will generalize the training components and the training process of GANs, so that it could be used as a starting point in building specific applications of GANs.
2. **Applications**: To use the framework to implement a set of popular applications of GANs.
3. **Exploration**: To explore new theoretical approaches to GANs by navigating the most promising approaches in literature, and implement a subset of these using the framework.

## Motivation
Here listed are some of the motivations behind this proposal:
- At the moment, there is **no other library** that provides a comprehensive and native framework for Generative Adversarial Networks.
- There is a **growing number of applications** that are being researched and implemented. Having a centralized, adaptable and extensible framework for developing GANs would be a great tool for AI/ML researchers and enthusiasts.
- There are popular Tensorflow and Pytorch implementations of specific applications (see #applications), but most GAN components (i.e. Generators, Discriminators and respective training processes) could and should be **standardized** in a framework that can be easily extended to these applications, facilitating their development end extension.

## Background information on GANs
Generative Adversarial Networks have recently become a very hot topic machine learning. While there is always some form of hype associated with new techniques in machine learning, GANs have shown exceptional results and pioneering applications in unsupervised learning. Backed by an groundbreaking paper by Ian Goodfellow, GANs have been recognized by the general ML community as a potentially upcoming breakthrough in the field. Yann LeCun, Facebook’s Director of AI Research, calls them the most important recent idea in machine learning. [link]

The basic idea behind GANs, as the name may suggest, is to have a training process set up as a game between two neural networks. We have a generator network G, and a second discriminator network D.

D tries to classify input samples as either coming from G or a model distribution representing the “real” data. The objective of G is to generate samples such that D classifies them as real data.

At the end of the training process, we will have a discriminator with good internal representation of the data (which we could use for data preprocessing tasks, like feature extraction), but more interestingly we will have a generator that is able to produce new samples that fit a particular data distribution.

GANs have shown incredible results in image synthesis, but there are promising theoretical applications to other fields of unsupervised learning.

## Proposed approach
### Deliverables
#### Framework
- A **general framework** that captures the common components and training processes of the Generative Adversarial Networks.
- On successive iterations of the framework we can extend this framework to support refinements to GANs, particularly the following:
    - [Stacked GANs](https://arxiv.org/abs/1612.04357)
    - [Wassertein GANs](https://arxiv.org/abs/1701.07875)
- On further iterations we can also explore related generative models:
    - [Variational Autoencoders (VAEs)](https://arxiv.org/abs/1312.6114)
    - [PixelRNN](https://arxiv.org/abs/1601.06759)

#### Applications
- Working implementation with the use of the framework of [Deep Convolutional GANs (DCGANs)](https://arxiv.org/abs/1511.06434) for image synthesis. The application will be able to generate novel images after training on various datasets
- Extend the DCGAN application to support [image inpainting/completion](https://github.com/bamos/dcgan-completion.tensorflow)
- Further extend to handle applications in [compressed sensing](https://github.com/AshishBora/csgm) (this paper and code has recently been published by my Data Science professor at UT, and I have had the chance to discuss this in details with him)

#### Exploration
Most of the exploration part of the project will focus on using GANs for Reinforcement Learning.
I have made some exploratory work we could use to leverage GANs and apply them to reinforcement learning. Convolutional neural networks work great in image recognition, NLP and recommender systems because these application have the property of translation invariance. I do believe we can apply the same concept to policy modeling as each action that an agent makes translates directly on the environment it is in, and some exploratory theoretical papers ([[1]](https://arxiv.org/abs/1606.03476), [[2]](https://arxiv.org/abs/1611.03852)) seem to suggest so as well.

I will be working on this specific problem over the next year as part of my final year undergraduate thesis/dissertation, and having the opportunity to probe theory and code using mlpack and its potential future GAN framework would be an honor.
It would also be great to collaborate with the student working on the “[Reinforcement learning](https://github.com/mlpack/mlpack/wiki/SummerOfCodeIdeas#reinforcement-learning)” project to share ideas and explore different concepts in the relationship between reinforcement learning and deep learning.

### Preparatory work
Leading to the official start date, I am planning on completing a few preparatory tasks. This guarantees that I am in the perfect position to start actively contributing to the organization and making meaningful progress on the project from Day 1.

- Style guide review
    - *Progress*: went through the design guideline documents, will get more practical familiarity with the design guidelines by exploring the codebase
- In-depth exploration of the mlpack’s ANN framework
    - *Progress*: reviewed the implementation of some classes (mostly FFN and RNN), as well as some layers and optimizers
- Active contributions to the repo (go through open/new issues and tackle the feasible ones):
    - *Progress*: So far, I made a minor PR request that was merged ([#966](https://github.com/mlpack/mlpack/pull/966))
- mlpack implementation of a GAN for basic probability distribution: Implement a basic Generative Adversarial Network that trains a Generator able to product data points in a Gaussian distribution, given no prior knowledge.
    - *Progress*: first implementation ([gan.hpp](https://github.com/gvsi/mlpack-gsoc/blob/master/gan.cpp)) can be found on this. There's only one small detail I need to figure in the training process (instead of calling `<model>.Train` which calls `FFN`'s optimizer, I just need to make a single forward and backward pass through the network).
- Initial analysis of the literature related to GANs
- General community bonding

### Proposed timeline
**April - May**: preparatory work

**May 30 - June 23**: Framework

**June 23 - June 30**: preparatory work for Applications (pre-planning, paper revisions)

*1st evaluation*

**June 30 - July 24**: Applications

*2nd evaluation*

**July 24 - August 6**: Exploration

**August 6 - August 29**: new iteration on deliverables, further work

*Final evaluation*

## Relevant experience and related work
I have extensive Artificial Intelligence and Machine Learning experience, gained through university coursework and independent projects/learning.

I have taken classes in AI and Reinforcement Learning, Machine Learning (both at university and Andrew Ng’s MOOC), a Data Science Lab (applied machine learning and deep learning), Data Mining, Natural Language Processing, Big Data Programming (Hadoop and Spark), and Computer Vision with a focus on top-down ML techniques.
I have a solid mathematical background in statistics, linear algebra, and calculus.

I also have relevant experience in Software development, including exposure to bigger scale projects and codebases, as well as familiarity with web development, mobile development, and several programming paradigms: Object-oriented programming (C++, Python, Java), functional programming (Haskell), declarative programming (Prolog).

Here are some relevant projects I had a chance to work on:
- [Face morphing](https://github.com/gvsi/face-morphing): a basic implementation of face morphing using DCGANs in Tensorflow
- [ClickBlock](https://github.com/gvsi/ClickBlock): a Chrome extension that works identifies clickbait article titles on your Facebook feed and replaces them with a more meaningful summary of the content
- [Ask](https://github.com/gvsi/ask): company I started with a classmate, acquired by the University of Edinburgh in 2015
- [HackProphecy](https://github.com/gvsi/HackProphecy): Tool that predicts with 94% accuracy hackathon winners using SVMs and NLP on scraped Devpost’s data
- [Digit classification](https://github.com/gvsi/digit-classification): Classification of MNIST data with different ML techniques
- [Netflix challenge](https://github.com/gvsi/netflix-challenge): an attempt at the Netflix challenge in C++ (achieved a RMSE of 0.97)
- [AI planning](https://github.com/gvsi/ai-planning): An AI robot car parking agent, written in Prolog using situational calculus
- [Allocator](https://github.com/gvsi/allocator): An implementation of C++ native allocator (coursework)
- [Game of Life](https://github.com/gvsi/game-of-life): a Game of Life implementation in C++ (coursework)

More projects could be found on my [Github account](https://github.com/gvsi).

## Discussion
I would love to hear what you think about this proposal! For any comment, request or critique, please create a [new Github issue](https://github.com/gvsi/mlpack-gsoc/issues/new) on this repository and we can take the discussion from there.

Thanks,

Gio
