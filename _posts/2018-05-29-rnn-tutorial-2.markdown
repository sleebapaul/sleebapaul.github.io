---
title: "Language Modelling using Recurrent Neural Networks (Part-2)"
layout: post
date: 2018-05-28 2:39 pm
image: /assets/images/artificial_intelligence.png
headerImage: true
tag:
- tutorial
- deep learning
star: true
category: blog
author: sleebapaul
description: Language modelling using basic RNN
mathjax: true
---

### Disclaimer

The audience is expected to have the basic understanding of Neural Networks, Backpropagation, Vanishing Gradients, and ConvNets. Familiarization of PyTorch is appreciated too, as the programming session will be on it.

# Introduction 

In the previous post we briefly discussed why CNN's are not capable of extracting sequence relationships. Fundamental reason for that failure is assumption of independence among the training and test examples. Say, if each image is a data point, then each image is considered as independent examples. Since they're independent, after a data point is processed, there is no purpose for storing state of the network. Why? 

Because you don't need that anymore. You saw the picture of a monkey, you identified it as a monkey and story is over. You don't need the monkey information to understand the **next image** of a piglet. (If you know a few things about CNNs you might have an objection here and the argument will be on parameter sharing. Parameters of CNNs are shared on spatial patterns as discussed in [previous post](https://sleebapaul.github.io/rnn-tutorial/){:target="_blank"}. CNNs can't carry temporal patterns. Later, we'll explain temporal parameter sharing of RNN in detail.)

<p align="center">
  <img src= "/assets/rnn_gospel_two/monkey_piglet.jpeg" />
</p>

But if data points are related in time, independence of data points is unacceptable. Frames from video, snippets of audio, and words pulled from sentences, represent settings where the independence assumption fails. We've listed such a bunch of problems in the first part of this tutorial too. 

In this post with we'll discuss Language modeling, a well-defined sequence problem and then dive deep into RNNs which can solve it. 

# Language Modelling 

In simple words, language modeling is generating the next **token** (it can be a character or  a word) in accordance with previous tokens (words or characters). 

In the previous post, we've seen many examples of it. 

1. Sleeba is native of Kerala. He can fluently speak _____. 
2. I bought my poodle from Paris. He barks _____.
3. I bought my poodle from Paris when I was staying with Sleeba. He has this trait of nodding while having food. But he loves __.

Here as you see, the past context is really important to predict the next token. If we miss that, then the entire sequence/sentence becomes meaningless. So the problem is defined. How to represent it in mathematical terms? 

![image-center](/assets/rnn_gospel_two/language_model_eqn.png){: .align-center}

Don't panic. We can explain that 😁 

It is as simple as,


```P(Is Trump linked with Stormy Daniels?) =```  
`P(Is) x`  
`P(Trump | Is) x `  
`P(linked | Is, Trump ) x`  
`P(with | Is, Trump, linked) x`  
`P(Stormy |  Is, Trump, linked, with) x `  
`P (Daniels | Is, Trump, linked, with, Stormy) x `  
`P (? | Is, Trump, linked, with, Stormy, Daniels)`

As we can see, probability of each word is a conditioned on previous words, thus the context is well maintained throughout the sentence. So if we can learn these probability distributions, then we can create meaningful language models. You might be thinking now as how weird this has become, right? Conditional probabilities, an equation takes after the iguana and a lot of mathematical mess... All are set up for representing something as easy as pie for us. That's the thing about homo sapiens. We are cool in many ways 😎

Mainly, we can do language modelling in two levels
1. *Character level* 
2. *Word level*

A language model will always have a **dictionary**, which is a collection of all the tokens that will be used in that model. The difference in above mentioned levels is that, their dictionaries are different. For example, the dictionary of a character language model of english will be the collection of 26 small letters, 26 capital letters, space and special characters of english. Dictionary size is not huge, still we can expect entire english vocabulary in the generative model. What about word level representation? We'll only be having vocabulary of training data in the dictionary. So we can't expect our generative model to produce a new word during testing. Same time, we may see gibberish words while testing a character level model. So the approach selection is a tradeoff upon our application. 

The simplest language model is randomly picking each character from its dictionary. The below code snippet depicts the simple character level language model. You may try a word level language model by yourself.

<script src="https://gist.github.com/sleebapaul/fa0a29a7acd6d6f85f2e4ee9d51d1156.js"></script>

Generated Sentence is, 

```W>X^Spz,wGOr(!C?uac-DqXvX_b^lv/S^p~cs)NjKWz;O+j"cZn jwZRK=I(xdD>tjgjF[BTc.mii`l<b/x#/,}(lIn\t":Ij&im```


Though this approach is simple to implement, we fail here to maintain the context. The probablity of every character generated is the same ($\frac{1}{Dictionary\ size}$), it is not conditioned on previous inputs. Thus the generated text is gibberish. Can we do better?

# Recurrent Neural Networks (RNNs)

RNNs are entirely different from usual neural networks when it comes to architecture. They are specially designed for processing sequential data, and they are not a new topic in history of DL either. In 1982, John Hopfield published a [paper](http://www.its.caltech.edu/~bi250c/papers/Hopfield-1982.pdf){:target="_blank"} in context of cognitive science and computational neuroscience which contained the idea of RNNs. From that paper to [Neural Machine Translation](https://github.com/tensorflow/nmt){:target="_blank"}, we've traversed a lot in this domain.   

Let's jump into the notations in order to understand the architecture of RNNs. 

## Notations

Imagine $x$ is the input sentence/sequence goes a black box named RNN and we get an output word $y$ for our language modelling problem.   

![image-center](/assets/rnn_gospel_two/blackbox_rnn.svg){: .align-center}

Using temporal terminology, an input sequence consists of data points $x^t$ that arrive in a discrete sequence of time steps indexed by $t$. 

Got it? I don't know why plain english is not a criteria, while explaining something scientific or mathematical 😂 Actually this is simple. For example, in the word level language modelling, sequence $I\ support\ LGBT\ rights$, 

$x^1$ = $I$,   
$x^2$ = $support$  
and so on.

While in character level, 

$x^1$ = $I$,  
$x^2$ = $\'\ \'$,  
$x^3$ = $s$,  
$x^4$ = $u$  
and etc. 

If you've noticed, we don't need `space` token for word level modelling since words are obviously seperated by space.

Now how a word/character is represented mathematically? There are many ways to do that and it is worth another blog post. Here we will breifly discuss two types. 

#### One hot vectors (Sparse representation)

Imagine your dictionary is `rights, human, LGBT, equality, I, support, a`. Vocabulary size is 7. Now, let $x$ be the input sentence `I support LGBT`, then representing `LGBT` in the sentence will be, 

$x^4\ =\ \begin{bmatrix}0 & 0 & 1 & 0 & 0 & 0  & 0\end{bmatrix}$  

What if the vocabulary size is $100000$. Then $x^4$ will be a $(1$ x $100000)$ matrix with a hell lot of zeroes. That's why it is sparse representation. As the dictionary size increases, the computational and memory cost increases. Moreover, it skips the relationship a word with other words, which is less intuitive. Read more about one-hot encoding at [here](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f){:target="_blank"}.

#### Word embeddings (Dense representation) 

Word embeddings cover up all the pitfalls of one hot vectors. First, their representation vector size don't increment with the vocabulary size like we just saw above. For almost all the embedding algorithm, the vector dimensions are fixed. Secondly, they are not just ones and zeros but decimals representing the relationship a word between other words. Thus they are dense representations.

For example, [GloVe: Global Vectors for Word Representation by Jeffrey Pennington, Richard Socher and Christopher D. Manning from Stanford](https://nlp.stanford.edu/projects/glove/){:target="_blank"}, converts the word "LGBT" to a 300 dimension vector. In a 300 dimension space, `LGBT` vector will be close to the 300 dimension vector of `human` to represent the relationship that LGBT community are also the part of homo sapiens. At the end of the day, the sample dense vector of `LGBT` will be, say,   

$x^4$=$\begin{bmatrix}0.2218 & 0.3812 & 0.8845 & ...\end{bmatrix}$  

Using these word embeddings, there are interesting relationships that can be learned like `King` + `Woman` = `Queen`, `India` to `New Delhi` is `Thailand` to `Jakartha` etc.  

I'm not going to explain how these decimals are generated. But from the above explanation, if you get the gut feeling that the black box RNN will be able to map context more intuitively with word embeddings than one hot vectors, then that's enough for this tutorial 😁 Though I highly recommend to read about embeddings at [here](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795){:target="_blank"}, [here](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12){:target="_blank"}, [here](https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c){:target="_blank"}, and [here](https://medium.com/swlh/playing-with-word-vectors-308ab2faa519){:target="_blank"} since wisdom is sexy.

## Architecture

Usually when we search for an RNN tutorial, we get the image given below. This is largely inspired by Christopher Olah's famous blog post on [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/){:target="_blank"}.

<p align="center">
  <img src= "/assets/rnn_gospel_two/rolled_rnn.png" />
</p>

A single RNN unit recursed to itself. This representation has a two rudimentary confusions. 

- This is a cyclic graph representation

Neural network graphs we've seen so far are acylic. Particularly, they are directed acyclic graphs (DAG). They start from the input nodes and reach the output nodes without loops. So? 

The mighty [back propagation is defined only for acyclic graphs](https://shapeofdata.wordpress.com/2016/04/27/rolling-and-unrolling-rnns/){:target="_blank"}. So we can't apply our learning algorithm in this RNN representation. Thus we've the unrolled architecture. 

![image-center](/assets/rnn_gospel_two/unrolled_rnns.png){: .align-center}

Or more intuitively, 

![image-center](/assets/rnn_gospel_two/rnn_sequence.svg){: .align-center}

Here the single cell is unrolled to a sequence of cells and made it acyclic. Now backpropagation is possible and it has a which has a fancy name too. Back Propagation Through Time (BPTT). But there is another issue. 

- We've learned all the neural network architecutures in **neuron** level, but this is a **cell** level explanation. 


In this post, I'm planning a neuron level explanation after introducing an RNN cell which is far more intuitive than a cell level implementation. 

To start with, let's define the equations of an RNN cell. 

**Equations**

$$
h^{[t]}\ =\ tanh(W_{hh} \cdot h^{[t-1]}\ +\ W_{xh} \cdot x^{[t]} + b_h)
$$

Let's begin with $h^{[t-1]}$. This is the hidden state input from previous RNN cell to current cell (Refer the diagram above).  

A hidden state? 🤷 Why we need a state now? We never had a state vector for CNNs. Why now? 

Remember we talked about maintaining context of a sentence? For this purpose, we need the information from past, right? Then $h^{[t-1]}$ is our man for that. I love to call $h^{[t-1]}$ the context vector rather than a hidden state vector since it carries the past context of the sequence.

Now let me show you the beauty of RNN architecture. Along with the current input $x^{[t]}$, we give this **context vector** from past to get the future. Perfect, ain't it? 💯

Now have a closer look at the equation. We've two weights, $W_{hh}$ and $W_{xh}$ which is going to adjusted or say **learned** while we train these cells with examples. What these weights are going to be learned? 

$W_{hh}$ will learn what needs to remembered or forget from past, that is from $h^{[t-1]}$. $W_{xh}$ learns about contribution of current input $x^{[t]}$. Together with both $W_{hh}$ and $x^{[t]}$ we will build our new context vector $h^{[t]}$ which has information from past and present. We're going to use this new context vector for two things.

Let's go to next equation for the first application. 

$$
y^{[t]}\ =\ W_{hy} \cdot h^{[t]}\ +\ b_y
$$

We're generating an immediate output using the current context h^{[t]} where W_{hy} learns about creating an output from current context. See, this is a completely optional decision. We can create an output anytime we would like to. Based on that, we can create different models for RNNs. The model we discussing now is `Many to Many (Synced)` model. There are other models too as shown in the figure below. This picture as well as many key ideas are taken from the bible of blog posts on RNNs. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/){:target="_blank"} by Andrej Karpathy. Love you Andrej ❤️


![image-center](http://karpathy.github.io/assets/rnn/diags.jpeg){: .align-center}

$$
o^{[t]}\ =\ softmax(y^{[t]})
$$

The third equation is not really a part of the architecture but we all know what a softmax does for us. I've not talked a lot about biases too, as they are pretty understood too. 

Second application of the context vector is to pass the baton to next time step. 

## Neuron Level representation 

So we've seen the insanely intuitive equations on an RNN cell. But let me unveil these cells in a neuron level so that you may understand it thoroughly. 

Let's go back to equation 1. 

$$
h^{[t]}\ =\ tanh(W_{hh} \cdot h^{[t-1]}\ +\ W_{xh} \cdot x^{[t]} + b_h)
$$

Let's break this equation down, 

What is $W_{hh} \cdot h^{[t-1]}$ ? It's a linear tranformation or a simple neural network layer we've seen in many normal neural networks. 

![image-center](/assets/rnn_gospel_two/rnn_detailed_2.svg){: .align-center}

Now what about $W_{xh} \cdot x^{[t]}$? Another linear layer.

![image-center](/assets/rnn_gospel_two/rrn_detailed_1.svg){: .align-center}


You must have got the idea of second equation too. Now, let me unveil the neuron level RNN cell to you, ladies and gentlemen... Pooooffff !!!

![image-center](/assets/rnn_gospel_two/rnn_detailed_3.svg){: .align-center}

> I think this my diagram is pretty self explanatory. 

> But it is little big. 

> Agreed. Then I'll confine it to the following. 

![image-center](/assets/rnn_gospel_two/rnn_block.svg){: .align-center}


> AAAAAAAGHHH !!! I'm feeling good. Really really good. I can die in peace now. 

> Okay, die well !!!

> Nope... Actually I was overreacting... a bit...

> A bit?

> 😬