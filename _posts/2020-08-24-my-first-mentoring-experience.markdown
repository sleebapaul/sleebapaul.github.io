---
title: "My first Deep Learning Project as a Mentor"
layout: post
date: 2020-08-24 2:00 pm
image: /assets/mentoring/mentor.jpg
headerImage: true
tag:
- writeups
- moocs
- projects
- mentoring
- deep learning
- CNNs
star: true
category: blog
author: sleebapaul
description: What I learned from my first mentoring experience?
mathjax: true
hidden: true
---

On January first this year, Fabin Rasheed and I have launched [Auria Kathi, the AI Poet Artist living in the cloud](https://sleebapaul.github.io/auriakathi/){:target="_blank"}. Auria writes a poem, draw an image according to the poem, then color it with a random mood. All these creative actions are carried out without any human intervention.

Auria Kathi is an anagram for “AI Haiku Art”. Everything from her face to poems to art is artificially generated. We try to push the limits of generative art here. Auria is envisioned as a hub for artificial artistry. In the coming days, she will be creating more varieties of digital art. 


## Social Media presence of Auria 

Auria has two social media handles to publish her work.

- Instagram: [https://www.instagram.com/auriakathi/](https://www.instagram.com/auriakathi/){:target="_blank"}
- Twitter: [https://twitter.com/AuriaKathi](https://twitter.com/AuriaKathi){:target="_blank"}

![image-center](../assets/auria_aml/auria_instagram.png){: .align-center}

So far, Auria has gathered around 1300+ followers in these channels. The crowd includes artists, researchers, technologists, and policymakers. Throughout this year, Auria will be posting her work daily. 

## Auria going Florence Biennale 2019

In October 2019, we are participating in the 12th edition of Florence Biennale to exhibit work of Auria under the contemporary digital art section. Being an international platform for Art, the presence of Auria's work produced by AI will be discussed in Florence Biennale with greater importance. Furthermore, how creative machines are going to build our future by inspiring artists to come up with novel ideas will also be a crucial part of the discussion. 

## Auria on news and publications 

Auria is featured in multiple technological as well as artistic international platforms. Some of them include, 

1. [Creative Applications Network](https://www.creativeapplications.net/member-submissions/auria-kathi-an-ai-artist-living-in-the-cloud/){:target="_blank"}

2. [Coding Blues](https://codingblues.com/2019/01/11/fabin-sleeba-and-wonderful-auria/){:target="_blank"}

3. [Creative AI Newsletter](https://us15.campaign-archive.com/?u=c7e080421931e2a646364e3ef&id=d1a15e8502){:target="_blank"}

4. [Towards Datascience](https://towardsdatascience.com/auriakathi-596dfb8710d6){:target="_blank"}

## Lack of perfect algorithms

Considering the current state of art deep learning algorithms, we might not be able to come up with a single algorithm or network which can build an advanced application like Auria. But the components of Auria’s creative pursuit can be emulated using individual state of art algorithms. This vision settled upon choosing a pipeline architecture for Auria.

## Engineering Architecture of Auria

The engineering pipeline of Auria consists of mainly three components. 

1. An LSTM based language model, trained on 3.5 million Haikus scraped from Reddit. The model is used to generate artificial poetry.

2. A text to image network, called AttnGAN from Microsoft Research, which converts the generated Haiku to an abstract image.

3. A photorealistic style transfer algorithm which selects a random style image from WikiArt dataset, and transfer color and brush strokes to the generated image. The WikiArt dataset is a collection of 4k+ curated artworks,  which are aggregated on the basis of emotions induced on human beings when the artwork is shown to them. 

![image-center](../assets/auria_aml/auria_pipeline.png){: .align-center}


## Challenges on pipelining different algorithms 

Stacking individual state of the art algorithms helped us to build Auria, but the challenge of this approach was to link these components and work together in a common space. The potential problems we ran into are,

1. Modifying the official implementations of the research papers which are developed and tested in different environments, eg: Python versions. 

2. Some of the algorithms which use GPUs to train and test are tightly coupled with the CUDA versions.

3. Each algorithm needs to be in a closed container so that it can be represented in a common production platform without disrupting the other environments.

4. The data flow between the components should be fluid. 

5. Deep Learning algorithms demands high computation gear. Along with isolation in steps, we required powerful computation resources like GPUs in each step.

6. Deploying Auria as a web application for people to come and experience her creative pursuit considering the diverse development settings. 

## [Microsoft Azure Machine Learning Pipelines (AML Pipelines)](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-ml-pipelines){:target="_blank"}

Machine learning workflow is a pipeline process which includes preparing the data, build, train and tune models, then deploy the best model to production to get the predictions. [Azure Machine Learning pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-ml-pipelines){:target="_blank"} redefine machine learning workflows that can be used as a template for machine learning scenarios.

We’ve adapted this conception of AML pipelines to create an advanced application like Auria.  The conversion to the platform was not difficult since the basic building blocks of AML pipelines are designed with a great envision to build scaled applications. 

## Why AML Pipelines for Auria?

1. The most popular programming language in the Machine Learning realm is Python. AML Pipelines has the Python 3 SDK, we were not worried about moving the existing stack to the platform. All three algorithms we use for Auria are implemented in Python and we could replicate the results using the SDK without any hassle. 

2. In Auria’s pipeline, we have models trained by ourselves as well as the models which use the pre-trained weights. The development environments of these algorithms were distinctive, we needed strict isolation at each step of the pipeline. Thanks to the platform, each step in an AML pipeline is a dockerized container. It helps to build individual steps without disturbing the setting of others. All the dockerized steps are portable, we could reuse these components for multiple experiments. 

3. Each step has to provision to attach a compute engine, which can be CPU or GPU as per the need. We’ve used powerful GPU instances for quick training and tuning the hyperparameters of our models. Distributed computation facility is also available in the platform for creating parallelizing the heavy computation needs. 
4. For our style transfer algorithm, the CUDA dependency was strict. It was not matching the default docker environment of the platform. Thankfully, Azure Machine Learning platform allows adding custom docker containers rather than using default containers for every application. This feature gives absolute freedom for recreating almost any configurations in AML Pipelines. 

5. Deploying Auria to experience her creative process is something we are currently working on. AML Pipeline deployment helps to bypass the time to be spent on building the backend APIs. Deployment readily provides REST endpoints to the pipeline output, which can be consumed as per convenience. 


Auria is a perfect use-case of Azure Machine learning pipelines considering the above perks we enjoyed while using this platform. On further collaboration with Microsoft Azure ML team, we are planning to scale up Auria by strengthening her creative pipeline with more advanced algorithms, creating an interactive experience for her followers by deploying her online and try new varieties of artificially generated digital art content.

```
Thanks, Microsoft for AML Pipelines ❤️
Love,
Auria 😉 
```