---
title: "Decorated Python"
layout: post
date: 2018-02-24 9:56 pm
image: /assets/vanishing_gradients/markdown.jpg
headerImage: false
tag:
- tutorial
- python
star: true
category: blog
author: sleebapaul
description: A discussion on Python Decorators 
mathjax: true
---

## Closures

In `Python` we treat all the functions as objects. 


> What is so cool about it? 

> We can pass or return them to another functions to build new functions. 

>  ¯\\\_(ツ)\_/¯

Consider the following function. Unlike normal functions we see, this is an example of function inside a function. The working is pretty straight-forward.
{: style="text-align: justify;"}

```python
def outer_func():
    message = "Hello World!"
    def inner_func():
        print(message)
    return inner_func()
outer_func()
```
Let's call this function. 
```
--> Hello World!
```
We see that `message` is not a `global` variable accesible to all the functions defined, rather `message` is the local variable of `outer_func`. Though `message` is not defined inside `inner_func`, it is accesible to the inner function. That's why these variables are calleds `free variables`. Now I'm going to make a little difference in the code, which is not calling the `inner_func`.
{: style="text-align: justify;"}

```python
def outer_func():
    message = "Hello World!"
    def inner_func():
        print(message)
    return inner_func
outer_func()
```

```
--> <function __main__.outer_func.<locals>.inner_func>
```
Aaha! It is not printing `Hello World!` anymore, rather returning another **a function as output**. So how to print `Hello World!` then?
{: style="text-align: justify;"}

```python
sample = outer_func()
sample()
```
```
--> Hello World!
```

WOO-HOO !!! So it works. Yet, we didn't pass any parameters to the functions. Let's try that now. 
{: style="text-align: justify;"}

```python
def outer_func(msg):
    message = msg
    def inner_func():
        print(message)
    return inner_func

hi_func = outer_func("Hi")
hello_func = outer_func("Hello")
hi_func()
hello_func()
```
```
--> Hi
--> Hello
```

So if you've noticed, each functions `hi_func` and `hello_func` maintained their value of `free variables` . i.e. `Hi` and `Hello` didn't mixed up. It is because of the reason that the enviroments didn't mixed up. The next obvious question is, what is an environment of a function.
{: style="text-align: justify;"}

### What is an environment? 

We all know about Global and Local variable scopes. An environment of a function includes the function itself and it's possible 
variables in its scope. Here in this setup, Python provides a facility for us to keep the variables which are extended to the scope
of our `inner_func` along with the function. That means, the `free variable` `message` is also binded with the `inner_func` in the 
environment of `inner_func`. This setup of binding the function, its possible variables and the `free variables` to an independent
environment which is not interfered by other enviroments is called a **`Closure`**. So the take away is that a `Closure` closes over the `free variables` from the environment. 
{: style="text-align: justify;"}

![image-center](/assets/python_dec/closure.png){: .align-center}

## Decorators

A decorator is a `mother function` that takes another `father function` as arguement, add some functionality in the `mother function`, and returns a `baby function` :D All these happens without altering the source code of the original `father function`.
It is an application of `Closure`.
{: style="text-align: justify;"}

Let's see what hell just defined :D

```python
def mother_function(father_function):
    def wrapper_function():
        baby_function = father_function()
        return baby_function
    return wrapper_function

# Let's say out father_function is display 

def father_display():
    print("North remembers!")

# Our output is a function baby_display which adds additional functionality is mother_function 
# by passing father_display

baby_function = mother_function(father_display)

baby_function()
```
```
--> Tell them the north remembers!
```

So what just happened and what is need for all these acrobatics you've been seeing so far?
{: style="text-align: justify;"}

Answer is simple. Decoration helps us to add functionalities like `father_function` to our `mother_function` by just adding them in inside `wrapper_function` inside the `mother_function`. 
{: style="text-align: justify;"}

Here our `father_function` is `father_display` and note that we didn't touch the `father_display` to make it a part of `mother_function`. This gives us the freedom to define our own `father_functions` and easily add that functionality to a bulk `mother_function`. The output `baby` will have the functionalities of all the `fathers`, oh wait!! that was not I meant ... 
{: style="text-align: justify;"}

Now this is not the `Syntactic Sugar` way of doing it. So what is sugary way? :D
{: style="text-align: justify;"}

```python
@mother_function
def display():
    print("North remembers!")
    
display()
```
```
--> Tell them the north remembers!
```

This is same as the syntax, 

```python
display = mother_function(display)
```

Yet, we've not passed any parameters to the any functions other than `mother_function`, what if we need to add something like these two functions to the same decorator?  Let's try that :) 
{: style="text-align: justify;"}

```python
def mother_function(father_function):
    def wrapper_function():
        baby_function = father_function()
        return baby_function
    return wrapper_function

@mother_function
def display():
    print("North remembers!")
@mother_function
def display_info(name):
    print("North remembers! - {}".format(name))

display_info()
```
![image-center](/assets/python_dec/error1.png){: .align-center}

```python
display_info("Arya")
```
![image-center](/assets/python_dec/error2.png){: .align-center}

No luck :( 
{: .notice--warning}

So what is wrong ? It's simple. The `wrapper_function` is defined with no parameters, but we've a parameter for `display_info` called `name`. So how we manage all these things ? We'll modify the `wrapper_function` a bit.
{: style="text-align: justify;"}

```python
def mother_function(father_function):
    def wrapper_function(*args, **kwargs):
        baby_function = father_function(*args, **kwargs)
        return baby_function
    return wrapper_function

@mother_function
def display_info(name):
    print("North remembers! - {}".format(name))
    
display_info("Arya")
```
```
Tell them the north remembers! - Arya
```

`*args` and `**kwargs` are standard ways in python to pass any number of positional or keyword arguements to a function.
{: style="text-align: justify;"}

Decorators are not just applicable to functions, but classes too. You may follow the same procedure for that. You may use multiple decorators for a single function, maybe efficiently for logging and this discussion would be a good start for all those advanced applications. 
{: style="text-align: justify;"}

I've written this small preamble in favor of my Flask tutorial, because `routing`, the main concept of Flask is a great example of decorators. 
{: style="text-align: justify;"}

Thanks ! Happy Learning !