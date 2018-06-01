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

In the previous post we briefly discussed why CNN's are not capable of extracting sequence relationships. We've also defined the problem we would like to solve using Recurrent Neural Networks and listed a bunch of sequence problems in real-life. In this post with we'll discuss Language modeling, a well-defined sequence problem and then dive deep into RNNs which can solve it. 

# Language Modelling 

Language modeling is simply generating the next character or word in accordance with previous words or characters. In the previous post, we've seen examples of it. 

1. Sleeba is native of Kerala. He can fluently speak _____. 
2. I bought my poodle from Paris. He barks _____.
3. I bought my poodle from Paris when I was staying with Sleeba. He has this trait of nodding while having food. But he loves __.

Here as you see, the past context is really important to predict the next word. If we miss that, then the entire sequence/sentence becomes meaningless. How to tranlate this to mathematical terms? 

![image-center](/assets/rnn_gospel_two/language_model_eqn.png){: .align-center}

Don't panic. We can explain that 😁 

It is as simple as,


```P(Did Trump fuck Stormy Daniels?) = P(Did) x P(Trump | Did) x P(fuck | Did, Trump ) x P(Stormy | Did, Trump, fuck) x P(Daniels |  Did, Trump, fuck, Stormy) x P (? | Did Trump fuck Stormy Daniels)```

As we can see, probability of each word is a conditioned on previous words, the context is well maintained throughout the sentence. So if we can learn these distributions, then we can create meaningful language models. You might be thinking now as how weird this has become, right? Conditional probability, an equation takes after the iguana and a lot mess. All of them are setup for representing something as easy as pie for us. That's the thing about homo sapiens. We are cool in many ways 😎

The simplest language model is randomly picking each character from a dictionary. The below code snippet depicts the simple character level language model. You may try a word level language model by yourself.

<script src="https://gist.github.com/sleebapaul/fa0a29a7acd6d6f85f2e4ee9d51d1156.js"></script>

Generated Sentence is, 

```W>X^Spz,wGOr(!C?uac-DqXvX_b^lv/S^p~cs)NjKWz;O+j"cZn jwZRK=I(xdD>tjgjF[BTc.mii`l<b/x#/,}(lIn\t":Ij&im```


Though it is simple to implement, we fail here to maintain the context. The probablity of every character generated is the same ($\frac{1}{Dictionary\ size}$), it is not conditioned on previous inputs. Thus the generated text is gibberish.  Can we do better? The answer is YES! 