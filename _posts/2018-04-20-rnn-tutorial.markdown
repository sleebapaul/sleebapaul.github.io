---
title: "Language Modelling using Recurrent Neural Networks (Part-1)"
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
description: A hello language model tutorial using Recurrent Neural Networks
mathjax: true
---

### Disclaimer
The audience is expected to have basic understanding of Neural Networks, Backpropagation, Vanishing Gradients and ConvNets. Familiarization of PyTorch is appreciated too, as the programming session will be on it.


# Motivation 

We've already achieved a lot of milestones in Deep Learning (DL). Still, calling it Artificial **Intelligence** is not appropriate since solving intelligence is a whole another ball game. But some leaps in DL give us hope that one day we'll solve intelligence, not necessarily with DL, but somehow we'll solve it. Such a leap is Google Translate, which supports 103 languages now. Around 18 months ago, [Google Translate moved from good old Statistical Machine Translation(SMT) to Neural Machine Translation(NMT)](https://ai.google/research/pubs/pub45610){:target="_blank"} and the results were captivating. 

There are two things which are remarkable about new Google Translate.

1. It solved a complex real-life sequence problem using DL.
2. It is an end-to-end DL application. 

### What is a Sequence problem? 

> Imagine you're given a sequence. Fill in the blank space. 

> 1, 2, 3, 4, _ 

> Oh come on, it's 5. Did Google solve THIS? 

> Nope. Let me ask you, how do know it is five? Why is it not six? 

> Are you dumb? It's consecutive numbers differ by 1. It's easy. 

> So there is a relation in that sequence and you found it. Great. Now try this. "Sleeba is native of Kerala. He can fluently speak _____. "

> Malayalam, dude. Are you fooling around with me? 

> How do you know? 

> That you're fooling around with me? 

> Nope :D.  How do you know it is Malayalam?

> Because sane people like me can understand that fact that there is a relationship with the language someone can fluently speak with their native place. Kerala speaks Malayalam. 

>Again a relationship. So you understood the relationship with the word Kerala.  Try this. 
"I bought my poodle from Paris. He barks _____"

>Loud, maybe? 

>Why not French? 

>You're mad. How can a dog speak french?

>So context changed from Paris to Poodle.  Try this.
 "I bought my poodle from Paris when I was staying with Sleeba. He has this trait of nodding while having food.  But he loves ____."

>I didn't get the context. Who is "he" here? Sleeba or Poodle? :/ 

>Welcome to real-life Sequence problems :D 

A sequence problem is defined through data points confined in time. It is the prediction of future with the help of patterns learned from past. As mentioned above, language is a perfect example of real-life sequence problems. 

In human beings, solving sequence problems is a continuous/online process.  Our sensory and motor data sequences are continuously streamed to the Neocortex,  most evolved part of a mammal's brain. Then Neocortex perpetually anticipates our future actions by processing these streams. This curious virtue of our brain gives us the gift of intelligence. So, solving a sequence problem is a step closer to solving intelligence. 

### What is end-to-end learning?

Usually, an end to end learning refers to omitting any hand-crafted intermediary algorithms and directly learning the solution of a given problem from the sampled dataset. 

### What is not end to end learning? 

Let's take an example of classifying apples and oranges. What we'll do to identify them? We'll extract some features, simple. 

Color : 

Apple is red or greenish red. Orange is, umm... orange maybe? 

Surface:

Apple surface is smooth. For orange it is bumpy. 

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#999;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#444;background-color:#F7FDFA;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#fff;background-color:#26ADE4;}
.tg .tg-88nc{font-weight:bold;border-color:inherit;text-align:center}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{font-weight:bold;border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg" style="margin: 0px auto;">
  <tr>
    <th class="tg-88nc">Fruit</th>
    <th class="tg-7btt">Skin</th>
    <th class="tg-88nc">Color</th>
    <th class="tg-88nc">.......</th>
    <th class="tg-7btt">Label</th>
  </tr>
  <tr>
    <td class="tg-c3ow">Apple 1</td>
    <td class="tg-c3ow">Smooth</td>
    <td class="tg-c3ow">Red</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">Apple</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Orange 1</td>
    <td class="tg-c3ow">Bumpy</td>
    <td class="tg-c3ow">Orange</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">Orange</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Apple 2</td>
    <td class="tg-c3ow">Smooth</td>
    <td class="tg-c3ow">Greenish Red</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">Apple</td>
  </tr>
  <tr>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">...</td>
    <td class="tg-c3ow">...</td>
  </tr>
</table>

Now we'll represent these features mathematically, train a classifier in many apples and oranges. Hopefully, the classifier learns the difference between apples and oranges, thus yield great prediction accuracy on new samples. 

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#999;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#444;background-color:#F7FDFA;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#fff;background-color:#26ADE4;}
.tg .tg-88nc{font-weight:bold;border-color:inherit;text-align:center}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-7btt{font-weight:bold;border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg" style="margin: 0px auto;">
  <tr>
    <th class="tg-88nc">Skin - Smooth</th>
    <th class="tg-7btt">Skin - Bumpy</th>
    <th class="tg-88nc">Color - Red</th>
    <th class="tg-88nc">Color - Greenish Red</th>
    <th class="tg-7btt">Color - Orange</th>
    <th class="tg-amwm">Label</th>
  </tr>
  <tr>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">1</td>
  </tr>
  <tr>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0</td>
  </tr>
  <tr>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">1</td>
  </tr>
  <tr>
    <td class="tg-baqh">...</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">...</td>
    <td class="tg-baqh">...</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">...</td>
  </tr>
</table>


But, there are a few problems with these hand driven features. 

1. For apples and oranges, we can select the features with our intuition, but what about a rocket trajectory regression? Or about gene sequencing? We need subject experts for each problem we solve to decide the vital features to be extracted.

2. Next question is, what if these intuitions can go wrong? What if there are features and patterns in the data which are more important than the selected ones? 

3. The mighty Homo Sapiens don't learn or predict this way. Homo sapiens can learn and make inferences from raw text or image or a mere smell, we don't need specific features. So this approach is far cry from **intelligence**. 

4. Any ML algorithm to date is as good as it's input data. There is no black magic. If the features we provide are vague, then classifier will be helpless.

Now, what if we can simply learn the features also, from the raw data? Then we won't miss out the hidden features in data. We don't need experts too. Then learning starts from scratch, which is more close to **intelligence**. This is why the end to end learning is important. 

# Problem definition

Let's begin with ConvNets. We all know that ConvNets work so well with images. But why it is such a success? An image is a spatial distribution of pixel values/numbers. 

![image-center](/assets/rnn_gospel/lincoln_pixel_values.png){: .align-center}

So every pattern in an image is spatially related. If an algorithm can represent and address those spatial patterns, it can understand a picture.  Convolutions exactly do the same. 

But what about a sentence? 

```
I support LGBT rights.
```

Is it spatially distributed? If so, the following sentences should be meaningful too. 

```
Support I rights LGBT.
LGBT support I rights.
Rights I LGBT support. 
```

None of them are meaningful. The only meaningful sequence is "I" followed by "support", next is "LGBT" and then "rights".  So the relationships are not spatial, but temporal/sequential.  Let's elaborate that node. 

1. Who is supporting LGBT rights? Me. 
2. What I'm supporting? LGBT rights. 

These answers are coming out of a meaningful sequential relationship between all those words in that sentence. If we try to represent a temporal distribution as a spatial distribution, we'll lose these temporal relationships in that distribution and thereby it's meaning. At this junction, an image gets different from a sentence. Thus, we need a new architecture which can capture those sequential relationships. Let's list a bunch of everyday sequence problems before we wrap up. 

1. Time series prediction (Weather forecast, Stock prices, ...)
2. Speech (Speech Generation and Recognition, Synthesis,  Speech to Text, ...)
3. Music (Music Generation, Synthesis, ...)
4. Text (Language modelling, Named Entity Recognition, Sentiment Analysis, Translation, ...)

In the next part of this tutorial series, let's discuss what are RNNs the basic building block of Google Translate, how they are used for capturing the sequential relationships and how to build a language model using RNNs.