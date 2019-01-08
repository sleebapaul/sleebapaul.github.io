---
title: "Auria Kathi  —  An artist in the clouds."
layout: post
date: 2019-01-08 9:00 pm
tag: 
  - projects
  - generative art
  - deep learning
  - LSTMs
  - GANs
image: /assets/auria/header.jpeg
headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "Auria Kathi  —  An artist in the clouds."
category: project
author: sleebapaul
externalLink: false
---

## What is art? Is it the unsaid? The unsettling?

The last few years have been very happening in the field of Generative/Procedural art. We have seen some of the exciting applications of this field hitting mainstream media — may it be generative architecture like the [Digital Grotesque](https://vimeo.com/74350367){:target="_blank"}, or the [AI generated paintings](https://www.forbes.com/sites/williamfalcon/2018/10/25/what-happens-now-that-an-ai-generated-painting-sold-for-432500/#5faf7225a41c){:target="_blank"} which sold for a bang or even simple apps which produce an artistic rendering of photographs using Neural Style Transfer like [Prisma](https://prisma-ai.com/){:target="_blank"}.

Generative art could be, in a broad sense defined as art generated using a set of instructions, usually using a computer. The art could be produced as a digital version, a physical version or as a combination of both. The definition of the field is still as broad as the definition of “Design” and many new forms of expressions have been brought under this title.

Last year, I and a friend of mine  — [Fabin Rasheed](https://www.linkedin.com/in/nurecas/){:target="_blank"} got together to talk about this field. I love to play with Machine Learning algorithms and Fabin loves Art and Design. We were conversing about how Instagram has become a portfolio website. Being known for original posts rather than shared content, Instagram seemed like the perfect place to showcase works by creatives and to create engagement. We were looking at some of the artists on Instagram and the idea struck us !— What if an artist, living in the cloud, posts regularly in Instagram — A robot, a machine, a piece of code which creates art regularly and posts on Instagram and keeps creating engagement.

This is how [Auria](https://www.instagram.com/auriakathi/){:target="_blank"} was born. [Auria Kathi](https://www.instagram.com/auriakathi/){:target="_blank"} is an anagram for “AI Haiku Art”. We started off trying to create a bot which continuously produced Haikus (to us this meant short poems). We wanted Auria to create poems which does not make complete sense in the beginning but has some meaning to it eventually.

**Some of Auria's poetry**
![image-center](/assets/auria/poems.jpeg){: .align-center}

Post this, we generated images based on the poems and finally coloured (styled) it with emotions from the poem and broke them into sets. For the curious people among you, the full technical details are given towards the end of this article. 

![image-center](/assets/auria/art.png){: .align-center}

Auria has now become a standalone bot which requires no maintenance — she keeps posting a poem and an artwork every day for one year and lives entirely on the cloud. So far, she has gathered up some followers and comments by humans as well as others like her ;). She has also started self-promotion.

```Auria is the First artist living completely in the cloud and with an Instagram portfolio. Her studio opened in Instagram on 01–01–2019.```

We also gave Auria a generated face. We tried to make it a generic, yet generated face. She lives!

Although Auria does not require any maintenance, we are continuously improving her. We are planning on creating better poetry, imagery and relations between them. We are also working on a chatbot which will respond to some of the comments and messages. Further down the line, Auria is envisioned as an Artificial Artist’s Studio. A hub for artificial artistry. We are planning to work on creating generated videos using Auria’s face giving her a voice and generated content to talk on. Who knows what’s in store for this little baby. She is the first of her kind!

Follow Auria here: [Auria on Instagram](https://www.instagram.com/auriakathi){:target="_blank"}


## Technical details
Auria uses three major algorithms to produce poems and art.

#### Language modeling

The first step is to generate the poetry which is a Language Modelling task. We fed around [3.5 million haikus](https://github.com/bfaure/hAIku){:target="_blank"} to train a Long Short-Term Memory (LSTMs) Network. Then the trained network is used for generating haikus. The code is written using PyTorch library. Google Colab is used for training.

Sample:

"It’s good as you can  
and pull it on that power  
and go home.  
Sorry."


#### Text to image

Next task is to convert the generated haiku into an image. We used the [Attentional Generative Adversarial Network (or AttnGAN)](https://arxiv.org/abs/1711.10485){:target="_blank"}, a paper by Microsoft Research in November 2017, which can generate output shapes from the input text. AttnGAN begins with a crude, low-res image, and then improves it over multiple steps to come up with a final image. Its architecture is a mix of GANs and Attention networks, which demand a multimodel optimization.

Since AttnGAN is a large network to train and our computation facilities were minimum, we used the pre-trained weights of the network which was originally trained in MS COCO dataset. The network can generate an output image of size 256x256. The sampling of AttnGAN is done in Google Colab.

Sample:

![image-center](/assets/auria/raw.png){: .align-center}  


#### Coloring the generated image

To bring in Auria’s mood and emotions, we transferred colors and shapes from sample images of the [WikiArt Emotions Dataset](http://saifmohammad.com/WebPages/wikiartemotions.html){:target="_blank"}. WikiArt Emotions is a dataset of 4,105 pieces of art (mostly paintings) that has annotations for emotions evoked in the observer. The pieces of art were selected from WikiArt.org’s collection for twenty-two categories (impressionism, realism, etc.) from four western styles (Renaissance Art, Post-Renaissance Art, Modern Art, and Contemporary Art). This study has been approved by the NRC Research Ethics Board (NRC-REB) under protocol number 2017–98, Canada.

The emotion images are picked at random, to attain diversity in Auria’s work. Additionally, [FastPhotoStyle by NVIDIA](https://github.com/NVIDIA/FastPhotoStyle/blob/master/TUTORIAL.md){:target="_blank"} is used for transferring the emotion image styles. Note that, existing style transfer algorithms can be divided into categories: artistic style transfer and photorealistic style transfer. For artistic style transfer, the goal is to transfer the style of a reference painting to a photo so that the stylized photo looks like a painting and carries the style of the reference painting. For photorealistic style transfer, the goal is to transfer the style of a reference photo to a photo so that the stylized photo preserves the content of the original photo but carries the style of the reference photo. The FastPhotoStyle algorithm is in the category of photorealistic style transfer. Images were generated using Google Colab. 

![image-center](/assets/auria/painted.png){: .align-center} 

The output colored image is scaled up the image to 1080x1080 using Photoshop to maintain quality.

Sample:

![image-center](/assets/auria/scaled.jpeg){: .align-center} 


## Face of Auria

We held on to the idea of artificiality throughout Auria. Thus the decision was taken to generate an artificial face for Auria. The quest for a generated face ended in [Progressively Growing GANs by NVIDIA](https://github.com/tkarras/progressive_growing_of_gans){:target="_blank"}, which is the most stable training schema for GANs to produce high-resolution output. Here she is :wink:

![image-center](/assets/auria/auria.png){: .align-center} 

## Final Thoughts

We conceived Auria as a flawed, temperamental, amateur artist. She has all those traits in her work and the studio she runs. The only difference is that she is not a physical being.

Added to that, art is all about interpretations. It’s a reflection of the beholder. So, here we are starting a new genre for looking at things with a few questions in our minds.

`Will artistry of algorithms add value to human life?`  
`Can Auria find a space between humans?`  
`Will she bring new meanings to this world without physically existing in it?`

We’re looking forward to the answers to these questions.

[Email Auria](auriakathi@gmail.com){:target="_blank"} || [Follow on Instagram](https://www.instagram.com/auriakathi/){:target="_blank"}  || [Follow on Twitter](https://twitter.com/AuriaKathi){:target="_blank"}

