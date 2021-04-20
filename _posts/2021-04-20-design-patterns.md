---
title: "A primer to design patterns in software development"
layout: post
date: 2021-04-20 2:00 am
image: /assets/mentoring/mentor.jpg
headerImage: true
tag:
- writeups
- design patterns
- software development
- tech
star: true
category: blog
author: sleebapaul
description: My first design pattern
mathjax: true
hidden: true
---

### Disclaimer

I know people who have been software engineers for years and still don't care/know about design patterns in software engineering. And that's completely okay. Though this post discuss in detail about one of the famous design patterns, I don't want to be the gatekeeper of so-called "best engineering practices" while writing software. If you don't use it, you don't use it. Anyway, these practices becomes inevitable when the codebase slowly takes the form a 20 ft. burmese python which is incredibly hard to maintain or scale. So it advised to refactor the code before it "smells".

# Design Patterns

Design patterns are templates or we could say a framework for solution to recurring/common problems that could happen in a software project. They are not exact solutions, rather something like a blueprint of how to approach the problem. More specifically, the design patterns deal with object oriented software design.

### Who invented the design patterns?

There are no inventors per se, when it comes to design patterns. If the same conflict occurs in multiple software projects, someone would address the issue by coming up with a pattern. Eventually, the pattern gets polished and revised by others, and finally, it becomes an established design pattern in the community. Having said that, there is actually an interesting story about the inspiration of origin of design patterns. The con­cept of pat­terns was first mentioned in "A Pat­tern Lan­guage: Towns, Build­ings, Con­struc­tion” by Christo­pher Alexan­der. This book had nothing to do with software engineering. Interestingly, the book dealt with designing urban environments. The idea was picked up by four authors named Erich Gamma, John Vlis­sides, Ralph John­son, and Richard Helm. In 1995, they pub­lished a book called "Design Pat­terns: Ele­ments of Reusable Object-Ori­ent­ed Soft­ware", which is considered to be the bible of design patterns.

### Why design patterns?

Interesting story. By why should someone use design patterns?

* **Software projects are susceptible to change**

Do you know a commercial software project that was written once and served forever? If so, the profession software developers would be in jeopardy. From the changes in services and subscriptions to API reformulation to hike in number of consumers, we have umpteen moving parts in a software project. Ideally, when a change is made, the existing codebase should not fall apart. But as we all know, it is not the case.

So, how to minimize the damages? There is no such silver bullet paradigm that can do it alone. Since, the design patterns are formulated based on the problems that could possibly happen in a project, by abiding the patterns, we can weed out the known problems and thereby minimize the damage. After all, if we've a proven solution, why bother starting from the scratch.

* **To make the best out of Object Oriented Programming (OOP)**

[OOP is widely accepted in the community for multiple reasons](https://stackoverflow.blog/2020/09/02/if-everyone-hates-it-why-is-oop-still-so-widely-spread/){:target="_blank"}. Design patterns make the best out of OOP by efficiently utilizing encapsulation , inheritance and polymorphism. If you can architect a OOP codebase that adheres design patterns, you definitely have a massive advantage comparing your peers. OOP with design patterns put forward the ideas of,

1. Encapsulate what varies <br />
2. Program to an interface not implementation <br />
3. Favor composition over inheritance <br />

* **The easiness in communicating high level architecture**

If two developers who knows design patterns well, it would be much easier for them to communicate the high level design of a project. Say, if you're designing a food delivery app's API backend, your API can choose Facade pattern, where a Facade class handles the queries from client and delegates the task to complex and interdependent backend functions. Now both of the developers have an idea about how the whole system is going to get built. Most of the times, individual programmers only see a tiny part of the codebase, thinking that designs don't exist.

<p style="text-align: center">
<img src="/assets/factory_pattern/facade_pattern.png" >
</p>


### Types of patterns







## References

[1] https://medium.com/@andreaspoyias/design-patterns-a-quick-guide-to-facade-pattern-16e3d2f1bfb6 <br />
[2] https://www.oreilly.com/content/5-reasons-to-finally-learn-design-patterns/ <br />
[3] https://sourcemaking.com/design_patterns <br />
[4] https://realpython.com/factory-method-python/ <br />

