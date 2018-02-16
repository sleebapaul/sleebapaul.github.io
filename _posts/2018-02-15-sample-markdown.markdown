---
title: "Mathjax Paditham"
layout: post
date: 2025-02-15 00:00
image: /assets/vanishing_gradients/markdown.jpg
headerImage: false
tag:
- tutorial
- deeplearning
star: true
category: blog
author: sleebapaul
description: A tutorial on Residual Networks
mathjax: true
---


---------------------------------------------------------

Let's test some inline math $x$, $$y$$, $$x_1$$ $$y_1$$.

Now a inline math with special character: $$\|\psi\rangle$$, $$x'$$, $$x^*$$.

Test a display math:
$$
   |\psi_1\rangle = a|0\rangle + b|1\rangle
$$
Is it O.K.?

Test a display math with equation number:

$$
\begin{align}
   |\psi_1\rangle = a|0\rangle + b|1\rangle \\
   |\psi_1\rangle = a|0\rangle + b|1\rangle \\
   how\ are\ you\ machane
\end{align}
$$

$$
\begin{align}
\frac{\partial J}{\partial W^2}\ = \ \frac{\partial J}{\partial A^3} * \frac{\partial A^3}{\partial Z^3} * \frac{\partial Z^3}{\partial A^2} * \frac{\partial A^2}{\partial Z^2} * \frac{\partial Z^2}{\partial W^2} \\\\
\frac{\partial J}{\partial W^2}\ =\ \frac{\partial (A^3\ -\ y)}{\partial A^3} * 1 * \frac{\partial (W^3 * W^2 * W^1 * X)}{\partial (W^2 * W^1 * X)} * 1 * \frac{\partial (W^2 * W^1 * X)}{\partial W^2} \\\\
\frac{\partial J}{\partial W^2}\ = 1 * W^3 * W^1 * X 
\end{align}
$$

$$ \begin{align} A \& = B \\ \& = C \end{align} $$

Ellam okay aayo ?
$$
   |\psi_1\rangle = a|0\rangle + b|1\rangle \\\\
   how\ are\ you
$$

Is it O.K. bro?

Test a display math with equation number:

$$
  \begin{equation}
    |\psi_1\rangle = a|0\rangle + b|1\rangle \\\\
    |\psi_2\rangle = c|0\rangle + d|1\rangle
  \end{equation}
$$




$$
  \begin{align}
    & |\psi_1\rangle = a|0\rangle + b|1\rangle \\\\
    & |\psi_2\rangle = c|0\rangle + d|1\rangle \\\\
    & Machan
  \end{align}
$$
Is it O.K.????

And test a display math without equaltion number:
$$
  \begin{align\*}
    \|\psi_1\rangle &= a\|0\rangle + b\|1\rangle \\\\
    \|\psi_2\rangle &= c\|0\rangle + d\|1\rangle
  \end{align\*}
$$
Is it O.K.??

Test a display math with equation number:
\begin{align}
    |\psi_1\rangle &= a|0\rangle + b|1\rangle \\\\
    |\psi_2\rangle &= c|0\rangle + d|1\rangle
\end{align}
Is it O.K.?

And test a display math without equaltion number:
\begin{align\*}
    |\psi_1\rangle &= a|0\rangle + b|1\rangle 
    \\\\
    \|\psi_2\rangle &= c|0\rangle + d|1\rangle
\end{align\*}
Is it O.K.?

----------------------------------------------------------
So, first we calculate gradient at third layer with respect to cost. 

\begin{align}
& \frac{\partial J}{\partial W^3}\ = \ \frac{\partial J}{\partial A^3} * \frac{\partial A^3}{\partial Z^3} * \frac{\partial Z^3}{\partial W^3} 
\end{align}
\begin{align}
& \frac{\partial J}{\partial W^3}\ =\ \frac{\partial}{\partial A^3}(A^3\ -\ y) * 1 * \frac{\partial}{\partial W^3} (W^3 * W^2 * W^1 * X)
\end{align}
\begin{align}
& \frac{\partial J}{\partial W^3}\ = 1 * W^2 * W^1 * X 
\end{align}


$$
  \begin{align\*}
    |\psi_1\rangle &= a|0\rangle + b|1\rangle \\\\
    |\psi_2\rangle &= c|0\rangle + d|1\rangle
  \end{align\*}
$$
----------------------------------------------------------