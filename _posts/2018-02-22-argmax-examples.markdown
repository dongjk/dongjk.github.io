---
layout: post
title:  "argmax examples"
date:   2018-02-22 02:25:50 -0800
categories: math symbol tips argmax
---
{% include mathjax.html %}

There have a lot simple and good interpretations for `argmax`, it return arg $x$ when function $f(x)$ is maximum, see below numpu code.
{% highlight python %}
import numpy as np
np.argmax([3,2,1])
#=> 0 #return the first index.
{% endhighlight %}

But when we reading papers and books, argmax usually come up with some complex formual which are not easy to understand if you are new to machine learning. Here are two examples. 

below formula is come from maximum like hood part, chapter 5, deeplearning book:

$$
\underset{\theta}{\arg \max} \prod_{i=1}^m P_{model}(x^{(i)};\theta)
$$
In this fomula, argmax will return $\theta$, which are weights when probability in learning mode is the maximum, here arg are $x^{(i)}$ and $\theta$, function is product of $P_{model}$

An other example, language translation section in sequence models of coursera deep leraning course, 

$$
\underset{y}{\arg \max} \prod_{t=1}^{T_y} P(y^{<t>}|x,y^{<1>},...,y^{<t-1>})
$$

Here $x$ is sentance before translate, $y$ is sentence after translate, $y^{<t>}$ is every word in it, e.g, we want translate `没有母牛关` to `there is no cow level`, $x$ is `没有母牛关`, $y$ is `there is no cow level` and $y^{<1>}$ is `there`, $y^{<2>}$ is `is`, $y^{<3>}$ is `no`, etc.

这里的argmax要返回一个句子,这个句子要使函数$\prod P$最大, 而这个h函数模型就是基于已有句子`there is no cow level`和已经翻译的部分情况下,找一个字满足最大的概率,比如已翻译部分是`没有母`那么下一个字是`牛`的概率$P(y^{<4>}|x,y^{<1>},y^{<2>},y^{<3>})$最大, 每个字的概率最大, 乘起来也是最大, 所以最终得到$y$为`没有母牛关`

In this example, argmax will return a sentence, which make function $\prod P$ maximum, this function is ask probability of next word with given sentence and alread tranlated part, e.g given sentence `没有母牛关` and already tranlated part like `there is no`, word `cow` will make probability $P(y^{<4>}|x,y^{<1>},y^{<2>},y^{<3>})$ maximum, and if every word have maximum probability, the product of those will be maximum, and finally get `there is no cow level`.