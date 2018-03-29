---
layout: post
title:  "Numpy axis 直观印象"
date:  2018-03-29 21:25:50 +0800
categories: math numpy
---

Axis在Numpy库的array操作中起了非常关键的作用, 它用于指示一个array(或者说tersor)的维度.

例如: 一个array的shape是(2,4,6), 那么第一个维度也就是axis=0对应的是shape 2, 第二个维度axis=1对应的是shape 4, 以此类推, 另外为了方便, axis=-1是值得倒数第一个维度.

这里有个动画可以直观感受下axis

![]({{ site.url }}/assets/article_images/2018-03-29-Numpy_axis_intuiation/s1.gif){:height="75%" width="75%"}

Numpy中的函数可以根据对dimension的操作分为下面几类
1. dimension保持不变
2. dimension减少
3. dimension增加
4. axis交换

下面的动画展示了Numpy是如何沿着axis进行保持和减少dimension操作的

![s2]({{ site.url }}/assets/article_images/2018-03-29-Numpy_axis_intuiation/s2.gif){:height="75%" width="75%"}

对于dimension增加和axis交换, 可以看下Numpy doc的例子, 动画太难做了┑(￣Д ￣)┍

|||
|:----------|:----------|
| [`expand_dims`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.expand_dims.html#numpy.expand_dims "numpy.expand_dims")(a, axis) | Expand the shape of an array. |
| [`swapaxes`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.swapaxes.html#numpy.swapaxes "numpy.swapaxes")(a, axis1, axis2) | Interchange two axes of an array. |
