---
layout: post
title:  "Numpy axis intuiation"
date:  2018-03-29 21:25:50 +0800
categories: math numpy
---


Axis play a key in Numpy array operations,
it indicate the dimension of an array(or a tensor), for example, if an array have shape (2,4,6)
first dimension is axis=0 coresponding to shape 2, second dimension is axis=1 coresponding to shape 4, etc, and for the convience, Numpy also can use axis=-1 to indicate the last dimension.

I made an animation for intuiation.


![]({{ site.url }}/assets/article_images/2018-03-29-Numpy_axis_intuiation/s1.gif){:height="75%" width="75%"}


Functions in Numpy to operate array can be classified by how to change dimensions.
1. keep dimensions
2. collapse dimensions
3. expand dimensions
4. switch axis

Here is a animation shows how to operate array along different axis when keep/collapse dimensions.

![s2]({{ site.url }}/assets/article_images/2018-03-29-Numpy_axis_intuiation/s2.gif){:height="75%" width="75%"}

For expand and sitch axis it is almost the same idea. Numpy doc example expain this very well.


|||
|:----------|:----------|
| [`expand_dims`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.expand_dims.html#numpy.expand_dims "numpy.expand_dims")(a, axis) | Expand the shape of an array. |
| [`swapaxes`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.swapaxes.html#numpy.swapaxes "numpy.swapaxes")(a, axis1, axis2) | Interchange two axes of an array. |
