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
hidden: false
---

I've been part of multiple Machine Learning (ML) projects since 2015. The list includes open-source side projects, collaborated works like [Auria Kathi](https://sleebapaul.github.io/auriakathi/){:target="_blank"}, and projects at work. 

But this year, I've unexpectedly bumped into a new role in my ML career. 

> Mentoring a Deep Learning (DL) project. 

I've helped people with their projects before as well. But oftentimes it is more or less like my opinion about someone's strategy to solve a problem using ML. Sometimes my inputs worked really well, sometimes it didn't. But this time things were a bit different. I've got to mentor a team throughout the journey of a DL project. 

This article is planned to describe what I've learned through the journey by citing some situations that happened in the project. That's how the reader would get an idea about both the project and mentoring. 

The following inferences are not just about leading a Machine Learning project, rather it can be applied to mentoring in general. Thus, the writeup is not deeply technical. Just that, people who are familiar with DL will be able to relate the examples in a better way. 
 
 
### Work with passionate people

It is not the first time people have approached me for a mentor role. Most of the time what happens is, mentees, lose their interest in the project as time goes by, and eventually, it becomes a zombie project. Since I've had a couple of bad experiences of this nature, I usually try to get devoid of mentor roles as much as possible. 

DL projects are so hyped nowadays, everyone wants a piece of it. But what these "AI Enthusiasts" don't understand is the elemental work of collecting, maintaining, and grooming the data is not so cool. It is a tedious job. 

When the team approached me last March, on the first call, I wanted the team to fix their data pipeline and get back to me. I didn't elaborate on anything about training models. 

This is a tactic I learned in a hard way. Because, time and time again, people just don't return once they are allotted with the primary but non-glamorous tasks. If they return, they are serious about what they are doing. 

This initial examination makes mentoring remarkably serene. Because tedious works test people. It reveals who they are. Passionate people don't just give up. 

> And guess what, this team had returned fixing their data pipeline. 

### Interruption is a tricky business

I'm not a big fan of micromanagement as it shrinks the room for self-improvement and ownership. But the tricky question for a mentor is when to intervene? 

If you are interfering with a task too often, the mentees will feel that it is micromanagement. Same time, setting loose will not yield desirable results either. This is a balance every mentor should find his/her own. Of course, it depends on the mentees and the situations. But there should be a balance. 

I'll give an example from the project. The project is an image classification task where image preprocessing is a requirement. My mentees were introduced to Python through DL. Sometimes, the lack of elemental programming knowledge in Python got them into trouble.

I wanted them to solve these issues on their own. But the solutions they've found for the problems made things more convoluted. It brought in numerous tight couplings, external dependencies, and easily breakable patches. Fast forwarding a series of such "hot-fixes", codebase became a serious mess. A point came where they were not able to fix problems without breaking some other parts. Then I realized that I could've intervened a little bit early to guide them with better approaches. It was avoidable mayhem. 

Sure, they've learned a lot from those mistakes, but from a project perspective, a mentor should intervene periodically before things get too much tangled. Also, this is an academic project. A tight timelined industrial project can't afford such slow learning experiments. 


### Judgments can go wrong

When everything works as expected, there will be no questions asked. But when things go wrong, we will start questioning everything. At some point, when nothing works, we will even start doubting ourselves. 

This is a common phase in almost all projects. We have a plan to get it working, and the plan fails miserably in practice. 

In our project, such an issue surfaced in the beginning. With limited computing resources, most of the teams will look into the Transfer Learning strategy when a new object detection or image classification problem pops up. We tried to do the same. 

Using a MobileNet architecture pre-trained in the Imagenet dataset, we changed the output layer and performed Transfer Learning. The results were pathetic. We tuned the hyperparameters, added more data ... but no luck. 

It was a testing moment for me as a mentor. From my experience, Transfer Learning is a reasonable method in such applications. My mentees were equally confused and looked at me for a resolution. I've had to find a valid reason why the approach didn't work. Then I dug deep into the problem. The findings were, 

1. The input data is not just images but 3D MRI scans of the brain. These scans are gone through a multi-level preprocessing and finally, the grayscale 2D image is created.

2. [Transfer learning from ImageNet models to large medical images has been very poor. We lose too much information going from high-resolution grayscale to low-resolution RGB which is required input for MobileNet pre-trained in ImageNet.](https://forums.fast.ai/t/transfer-learning-for-medical-radiography/4931/9){:target="_blank"}

This finding changed the whole strategy of the project. We've moved to a small network which could be trained with the limited computation facility we've, still gives great results. 

So, it's okay to be wrong. You're the mentor, not omnipotent. When things go wrong, accept it with an open mind, find what is going wrong, and adapt. Don't take the failures personally. If you do, then you'll find reasons to blame someone else or yourself and eventually become hopeless. Rather, find surprises as an opportunity to grow. 

> Sounds a bit dramatic, but it is a fact. 

### Stick with the success of the project 

You can't get away from the ideological disputes if you are actively involved in a project. Sometimes, people will disagree to agree. And it is okay. You're the mentor doesn't mean that you are right about everything. 

> Don't be a gatekeeper. 

Don't let your ego affect the success of the project. As a mentor, your priority is the completion of the project. It is not about a one-man show of all your ideas and knowledge. You're a hero, if and only if the project is a success. Otherwise, it is just a tool for feeding your ego. Don't be that guy who stops co-operating as his/her ego is hurt. 

### Be empathetic and patient

As a mentor, you are the one who is more exposed to the technology stack and best practices than your mentees. But people forget this fact at times.  Your mentees don't have the expertise you've. If they do, they won't be needing you as a mentor. Always keep this in mind. 

Expect mistakes from the team. Things that look stupid for you may not be alarming for them. So, when they make mistakes, be empathetic. Help them correct it iteratively. But never allow the same mistake to happen twice. 

Also, avoid shouting as much as possible. Not only that it doesn't solve the issue, no one goes home happy after a callous conversation. 

> And have patience. Good things will take time. 

During the project, I've lost my cool once and it was about documenting the results properly. The team rectified it immediately and it was in a better format.  But when I think about the incident now, I feel that I could've handled the situation much better. 

### Set templates

This is the best thing you can do as a mentor. Roll up your sleeves and set a basic infrastructure for your mentees. This approach helps them in multiple ways.

1. The mentees will have a lookup to refer to and start with. Most of the time, starting right is the biggest challenge for a noobie. 

2. On setting up a template, you've control over what you're expecting. You can easily detect if mentees derail far from the original architecture and bring them back. 

3. Implicitly, you will help the mentees to understand why you've chosen such an architecture and how it is going to help them while doing the project. On the next project, they are more likely to apply these best practices and infrastructure as they've already tasted the fruit of it. 

For this project, we had to do error analysis, try different datasets, do hyperparameter tuning,  and explore different network architectures. So I've set them a template that contains modules that can be easily reused for various experiments. I've convinced the team about the plug and play feature of these modules and encouraged them to write code in a modular manner with minimum external dependencies and coupling. It helped them to do faster experiments without breaking anything else. Now they are more likely to reuse that design paradigm in their future projects since they already know why such an approach was taken. 

Furthermore, I've introduced them to version control, better coding practices, and documentation tricks. Once the project got over, they were familiarised with a lot of topics other than Deep Learning.

### Give appreciation. Period. 

People love getting an acknowledgment of the work they do. If you like someone's work, tell them. Don't pretend to be a tough boss. 

When appreciating, try to appreciate the attitude than the achievement itself. For example, in our project, at the beginning our model was overfitting and we required more data. But, collecting MRI Scans from the ADNI website was not straight forward. The team actually spend a lot of time doing that tedious task. On each iteration, the model was fed with more data. Slowly we tackled the overfitting issue. The team had collected almost 1500 patients data for improving the classifier. 

When the validation accuracy reached 90+ percentage, we have appreciated the tenacity of team members to bring in the data rather than the achievement of reaching 90+ percentage accuracy itself. Appreciating the attitude helps people to understand the productive traits in them and possibly nurture those values for future endeavors.  

One more thing to be noted is overdoing appreciation. Appreciation should be earned. It can't be a Dopamine generating exercise. Once we have such an attitude, the trap of mediocre work can be shunned. A mentor should have an instinct to judge when to appreciate and when to push the mentees forward. 

## Final Thoughts

There are no cardinal rules for mentoring. Rules changes with situations and mentees. But there are some core values that can ease the process. I hope you may reflect on my findings, and let me know yours. 

If you would like to take a deeper dive into our project, have a look at the following links.

GitHub Repo: https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning
Medium article: 