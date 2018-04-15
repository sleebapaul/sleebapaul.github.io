---
title: "PyThesaurus"
layout: post
date: 2018-04-15 2:10
tag: 
  - projects
  - python
image: /assets/pythesaurus/python_logo.png
headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "This python package gets you the thesaurus of an inputted word."
category: project
author: sleebapaul
externalLink: false
---

Description
------------

This python package gets you the thesaurus of an inputted word from the best dictionary sites available online. 

Why you need this package?
---------------------------

Though python provides lexical resources like WordNet, the variety it can provide is poor. The rich content the [Thesaurus.com](http://www.thesaurus.com){:target="_blank"} or the [Dictionary.com](http://www.dictionary.com/){:target="_blank"} provides will help the user to enhance their approaches when he/she is dealing with text mining, NLP techniques and much more.

How to install? 
---------------

Use `pip` to install this library.  

```python
pip install py_thesaurus
```

How to use PyThesaurus?
-----------------------

**From python shell** 

```python
   from py_thesaurus import Thesaurus

   input_word = "dream"

   new_instance = Thesaurus(input_word)

   # Get the synonyms according to part of speech
   # Default part of speech is noun

   print(new_instance.get_synonym()) 
   
   print(new_instance.get_synonym(pos='verb'))

   print(new_instance.get_synonym(pos='adj'))
   
   # Get the definitions 

   print(new_instance.get_definition())

   # Get the antonyms 

   print(new_instance.get_antonym())
```

**From command line**

- Positional arguments

```
word --> Word to get definition/synonym/antonym for.
```

- Optional arguments
```
  -h or --help       Show this help message and exit
  -d                 get definition
  -s {noun,verb,adj} get POS specific synonyms
  -a                 get antonyms
```
- Command
```
   py_thesaurus [-h] [-d] [-s {noun,verb,adj}] [-a] word

   py_thesaurus -d -s verb -a dream
```

Contact
--------

1. PyPI link: https://pypi.python.org/pypi/py-thesaurus

2. Bitbucket: https://bitbucket.org/redpills01/py_thesaurus.git

3. Issue tracker: https://bitbucket.org/redpills01/py_thesaurus/issues                          

4. Email: redpillsworkspace@gmail.com   

_Made with Love by Redpills :heart:_