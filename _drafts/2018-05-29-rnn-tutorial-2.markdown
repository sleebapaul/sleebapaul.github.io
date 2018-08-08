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

In the previous post we briefly discussed why CNN's are not capable of extracting sequence relationships. Fundamental reason for that failure is assumption of independence among the training examples.  

Say, if each image is a data point, then each image is considered as independent example, right? 

But in the cases like frames from video, snippets of audio, and words pulled from sentences, this independence assumption fails. Because they have data points related in time and thus they are named as sequence problems. We need a new architecture which can capture this temporal relationship in sequence problems. 

In this post with we'll discuss Language modeling, a well-defined sequence problem and then dive deep into RNNs which can solve it. 

# Language Modelling 

In simple words, language modeling is generating the next **token** (it can be a character or  a word) in accordance with previous tokens (again words or characters). 

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

As we can see, probability of each word is a conditioned on previous words, thus the context is well maintained throughout the sentence. So if we can learn these past probability distributions, then we can generate the next word accordingly.  

You might be thinking now, how weird this has become, right? Conditional probabilities, an equation takes after an iguana and a lot of mathematical mess... All are set up for representing something as easy as pie for us. That's the thing about homo sapiens. We are cool in many ways 😎

We can do language modelling in two levels  

- *Character level* which generates a character at a time. 

- *Word level* which generates a word at a time. 

### Dictionary

A language model will always have a **dictionary**, which is a collection of all the tokens that will be used in that model. For example, the dictionary of a character language model of english will be the collection of 26 small letters, 26 capital letters, space and special characters of english. Dictionary size is small, still we can generate entire english vocabulary.  

What about word level representation? Dictionary will contain all the words present in our training data. But we can't expect our model to generate a new word which is not in our vocabulary. Same time, word level model are less gibberish than a character model since, a word is the basic building block rather than character.

### Simplest language model

The simplest language model is randomly picking each character from its dictionary. The below code snippet depicts the simple character level language model. You may try a word level language model by yourself.

<script src="https://gist.github.com/sleebapaul/fa0a29a7acd6d6f85f2e4ee9d51d1156.js"></script>

Generated Sentence is, 

```W>X^Spz,wGOr(!C?uac-DqXvX_b^lv/S^p~cs)NjKWz;O+j"cZn jwZRK=I(xdD>tjgjF[BTc.mii`l<b/x#/,}(lIn\t":Ij&im```


Though this approach is simple to implement, we fail here to maintain the context. The probablity of every character generated is the same ($\frac{1}{Dictionary\ size}$), it is not conditioned on previous inputs. Thus the generated text is gibberish. Can we do better?

# Recurrent Neural Networks (RNNs)

RNNs are entirely different from usual neural networks when it comes to architecture. If CNNs are best for spatially distributed data, RNNs are specially designed for processing sequential data. But they are not a new topic in the Deep Learning scene. In 1982, John Hopfield published a [paper](http://www.its.caltech.edu/~bi250c/papers/Hopfield-1982.pdf){:target="_blank"} in context of cognitive science and computational neuroscience which contained the idea of RNNs. From that paper to [Google Duplex](https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html){:target="_blank"}, which can take an appointment for us by buiding a reliable conversation with a barber, we've traversed a lot in this domain.   

Let's jump into the notations in order to understand the architecture of RNNs. 

### Notations

Imagine $x$ is the input sentence/sequence goes a black box named RNN and we get an output word $y$ for our language modelling problem.   

![image-center](/assets/rnn_gospel_two/blackbox_rnn.svg){: .align-center}

Using temporal terminology, an input sequence consists of data points $x^t$ that arrive in a discrete sequence of time steps indexed by $t$. 

Got it? I don't know why plain english is not a criteria, when it comes to explaining something scientific or mathematical 😂 Actually, this is a simple concept. For example, in the word level language modelling, sequence 

<p align="center" ><b>
I support LGBT rights
</b></p>

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

Next question is, how to represent this word/character mathematically? There are many ways to do that and it is worth another blog post. Here we will breifly discuss two types. 

#### One hot vectors (Sparse representation)

Imagine your dictionary is [`rights, human, LGBT, equality, I, support, a`]. Vocabulary size is 7. Now, let $x$ be the input sentence `I support LGBT rights`, then representing `support` in the sentence will be, 

$x^2\ =\ \begin{bmatrix}0 & 0 & 0 & 0 & 0 & 1 & 0\end{bmatrix}$  

What if the vocabulary size is $100000$. Then $x^2$ will be a $(1$ x $100000)$ matrix with a $1 and a hell lot of zeroes. That's why it is sparse representation. As the dictionary size increases, the computational and memory cost of one-hot representation increases. But most importantly, it skips the relationship between words, which is less intuitive. Read more about one-hot encoding at [here](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f){:target="_blank"}.

#### Word embeddings (Dense representation) 

Word embeddings cover up all the pitfalls of one hot vectors. They are learned using unsupervised methods like autoencoders.  

Firstly, their representation vector size don't increment with the vocabulary size. For almost all the embedding algorithms, the vector dimensions are fixed. 

For example, [GloVe: Global Vectors for Word Representation by Jeffrey Pennington, Richard Socher and Christopher D. Manning from Stanford](https://nlp.stanford.edu/projects/glove/){:target="_blank"}, converts the word "LGBT" to a 300 dimension vector. In a 300 dimension space, `LGBT` vector will be close to the 300 dimension vector of `human` to represent the relationship that LGBT community are also the part of homo sapiens. At the end of the day, the sample dense vector of `LGBT` will be, say,   

$x^3$=$\begin{bmatrix}0.2218 & 0.3812 & 0.8845 & ...\end{bmatrix}$  

Secondly, they are not just ones and zeros but decimals representing the relationship a word between other words. Thus they are dense representations.
Using these word embeddings, interesting relationships that can be learned like `King` - `Man` + `Woman` = `Queen`, `India` to `New Delhi` is `Thailand` to `Jakartha` etc.  

I'm not going to explain how these decimals are generated. But from the above explanation, if you get the gut feeling that the black box RNN will be able to map the context more intuitively with word embeddings than one hot vectors, then that's enough for this tutorial 😁 But I highly recommend to read about embeddings at [here](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795){:target="_blank"}, [here](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12){:target="_blank"}, [here](https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c){:target="_blank"}, and [here](https://medium.com/swlh/playing-with-word-vectors-308ab2faa519){:target="_blank"} since wisdom is sexy.

## Architecture

Usually when you search for an RNN tutorial, you get the image given below. This is largely inspired by Christopher Olah's famous blog post on [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/){:target="_blank"}.

<p align="center">
  <img src= "/assets/rnn_gospel_two/rolled_rnn.png" />
</p>

A single RNN unit recursed to itself. But, this representation has a two rudimentary confusions. 

- It is a cyclic graph representation

All the feed forward neural network graphs we've seen yet are acylic. CNNs structures we've seen are acyclic. Particularly, they are directed acyclic graphs (DAG). They start from the input nodes and reach the output nodes without loops. Why this is important?  

The mighty [back propagation is defined only for acyclic graphs](https://shapeofdata.wordpress.com/2016/04/27/rolling-and-unrolling-rnns/){:target="_blank"}. So we can't apply our learning algorithm in this RNN representation. Thus we've the unrolled architecture. 

![image-center](/assets/rnn_gospel_two/unrolled_rnns.png){: .align-center}

Or more intuitively, 

![image-center](/assets/rnn_gospel_two/rnn_sequence.svg){: .align-center}

Here the single cell recursed to itself, is unrolled to a sequence of cells. This is an acyclic architecture. Now backpropagation is possible and it has a fancy name too. Back Propagation Through Time (BPTT). 

But there is the second issue. 

- We've learned all the neural network architecutures in **neuron** level, but this is a **cell** level explanation. 

In this post, I'm planning a neuron level explanation of an RNN cell. To start with, let's define the equations of an RNN cell. 

#### Equations

Let's begin with $h^{[t-1]}$, the hidden state input from previous RNN cell to current cell (Refer the diagram above).  

- $$
h^{[t]}\ =\ F(W_{hh} \cdot h^{[t-1]}\ +\ W_{xh} \cdot x^{[t]} + b_h)
$$


A hidden state? 🤷 Why we need a state now? We never had a state vector for CNNs. Why now? 

Remember we talked about maintaining context of a sentence? For this purpose, we need the information from past. $h^{[t-1]}$ is our guy who carries the baton of past. I love to call $h^{[t-1]}$ the context vector rather than a hidden state vector since it carries the past context of the sequence.

Now let me show you the beauty of RNN architecture with this equation. **Along with the current input $x^{[t]}$, we give this context vector ($h^{[t-1]}$) from past to get the future ($h^{[t]}$) of the sequence.** Perfect, ain't it? 💯

Now have a closer look at the equation. We've two weights, $W_{hh}$ and $W_{xh}$ which is going to adjusted or say **learned** while we train these cells with examples. What these weights are going to learn? 

$W_{hh}$ will learn what needs to remembered or forget from past, that is from $h^{[t-1]}$. $W_{xh}$ learns about contribution of current input $x^{[t]}$. Together with both $W_{hh}$ and $W_{xh}$ we will build our new context vector $h^{[t]}$ which has information from past and present. We're going to use this new context vector for two things.
 
Let's go to next equation for the first application. 

- $$
y^{[t]}\ =\ W_{hy} \cdot h^{[t]}\ +\ b_y
$$

We're generating an immediate output using the current context $h^{[t]}$ where $W_{hy}$ learns about creating an output from current context. See, this is a completely optional decision. We can create an output anytime we would like to. Based on that, we can create different models for RNNs. The model we discussing now is `Many to Many (Synced)` model. There are other models too. Have a look at the figure given below. 

![image-center](http://karpathy.github.io/assets/rnn/diags.jpeg){: .align-center}

This picture as well as many key ideas are taken from the bible of blog posts on RNNs. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/){:target="_blank"} by Andrej Karpathy. Love you Andrej ❤️

- $$
o^{[t]}\ =\ softmax(y^{[t]})
$$

The third equation is not really a part of the architecture but we all know what a softmax does for us. I've not talked a lot about biases too, as they are pretty understood too. 

Second application of the context vector is to pass the baton to next time step.

> Wait, what is F?     

> Oh! I forgot that. F is the our good old non-linear activation function from feed forward networks. Like tanh, ReLU etc.    

> You're careless 😏    

> Oh! Come on 🙄    

#### Neuron Level representation 

We've seen the insanely intuitive equations on an RNN cell. But let me unveil these cells in a neuron level so that you may understand it thoroughly. 

Let's go back to equation 1. 

- $$
h^{[t]}\ =\ tanh(W_{hh} \cdot h^{[t-1]}\ +\ W_{xh} \cdot x^{[t]} + b_h)
$$

Let's break this equation down, 

What is $W_{hh} \cdot h^{[t-1]}$ ? It's a linear tranformation or a simple neural network layer we've seen in many normal neural networks. 

![image-center](/assets/rnn_gospel_two/rnn_detailed_2.svg){: .align-center}

What about $W_{xh} \cdot x^{[t]}$? Another linear layer.

![image-center](/assets/rnn_gospel_two/rrn_detailed_1.svg){: .align-center}


Now, you must have got the idea of second equation too. On that node, let me unveil the neuron level RNN cell to you. Brace yourself, ladies and gentlemen... Pooooffff !!!

![image-center](/assets/rnn_gospel_two/rnn_detailed_3.svg){: .align-center}

Current input and past context vector are linearly transformed and summed together. This sum is then passed to a $tanh$ to inject non-linearity/activation. Thus current context vector is generated, which is used for generating an optional output and pass the context to next time step. As simple as that !

> I think this my diagram is pretty self explanatory. 

> But it is little big. Christopher Olah chose the simpler diagram for a reason 🤔

> Agreed. I'll confine it to the following. 

![image-center](/assets/rnn_gospel_two/rnn_block.svg){: .align-center}


> AAAAAAAGHHH !!! I'm feeling good. Really really good. I can die in peace now. 
x
> Okay, die well !!!

> Nope... Actually I was overreacting... a bit...

> A bit?

> 😬

__Note__ 

People have confusion when the hyper parameter `hidden size` and `sequence length` of RNNs are discussed. Now from the neuron level diagram, we can clearly understand that, `hidden size` is number of neurons present in the hidden layer. Here it is three.   

`output size` can be sometimes referred as the `hidden size` as RNN has two outputs i.e. $y$ and $h^{[t]}$. As $y$ is optional, sometimes we treat $h$ as output. For good models, provided we've enough data(say, 100 million characters) we can afford to use a large `hidden size`. For small data samples (< 10 million characters) use small `hidden size` values.  

`sequence length` is the **number of RNN cells** unrolled/considered at once or simply the number of time steps. 

Now we'll discuss two most important concepts of RNNs and wrap up the talk and jump to code since,

> Talk is cheap. Show me the code - Linus Torvalds

## Parameter Sharing 

Let's talk about the parameter sharing of Neural Networks in general and understand the parameter sharing of RNN in a comparitive level. 

There is no parameter sharing in normal feed-forward networks. Every layer has a bunch of individual weights and biases and there are learned/updated on backpropagation.    

But when it comes to CNNs, we dramatically reduce the number of parameters using filters. Number of parameters are defined from size and depth of filter. These parameters are used for representing patterns. Thus, for representing different patterns, these parameters can be reused or say `shared`. Sharing helps to reduce the numbers parameters. Same idea is used in RNNs too, but in a slightly different way.  

CNNs share parameters for representing spatial features. RNNs does it through time for imbibing temporal features. The same weights are updated all time steps. 

> What does that mean? 🤔   

> Let me explain 😊

First of all, don't confuse cells with layers. Don't confuse time steps with layers. Multiple layer RNNs look something like this. 

![image-center](/assets/rnn_gospel_two/rnn_layers.png){: .align-center}

The weights $W_{hh}$ , $W_{hx}$ and $W_{hy}$ are updated at every time steps in a single layer. Since we update the weights in this manner, 


## Backpropagation Through Time (BPTT)



#### Vanishing Gradient Problem

