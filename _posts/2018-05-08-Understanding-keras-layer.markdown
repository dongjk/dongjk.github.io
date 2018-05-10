---
layout: post
title:  "Understanding keras layer"
date:   2018-05-08 09:15:50 +0800
categories: framework keras
---

Keras have a bunch of high level layers which very convenient to create variance of 
models, this article describe two things :
1. the concept and design of keras layer
2. how keras layer mapping to tensorflow backend

### Layer
Usually two steps to create a layer,
1. initialize an instance by run `__init__()` method.
2. call the instance by run `__call__()` method, this is the main step to make a keras layer.

like below:

{% highlight python %}
densor=Dense(256,activation="relu") #__init__()
d1=densor(merged)                   #__call__()
{% endhighlight %}

or

{% highlight python %}
d1=Dense(256,activation="relu")(merged) #__init__().__call__()
{% endhighlight %}



There are three parts for `__call__()`
#### Input tensor
The parameter pass to it, it can be tensors from previous layers, or initial input palceholders.
if it is initial input palceholders, `__init__()` method will initialize a tf placeholder and wrap it as a keras input tensor.
{% highlight python %}
            input_tensor = K.placeholder(shape=batch_input_shape,
                                         dtype=dtype,
                                         sparse=self.sparse,
                                         name=self.name)
{% endhighlight %}
#### Output tensor
The return value of `__call__()` method, output tensor(s) is calculated from input tensor,the calculate logic is defined in `call()`, every layer should implement a `call()` method to calculate it's output layer, if you customize a layer yourself, this the most important method need to implement.
Here we use Dense layer as a example see how it works:
{% highlight python %},
    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        ...
        ...
        return output
{% endhighlight %}
it simply calculate output by dot product input tensor with weights(here is self.kernel), exactly behavior as a dense layer to do.
#### layer instance itself. 
Layer itself is not a tensor, it holds weights which need by tensor operations, and also hold the logic to do operations.
every layer have a `build()` methods, if you customize a layer yourself, and this layer have weights, then you need `build()` method.
Use the Dense layer as example, `build()` function doing something like:
{% highlight python %},
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
{% endhighlight %}
it create a weight for Dense layer. and for upper example, we can see the weight are used by dot product with input tensor.


###Container
OK, now we have a layer with input tensor and output tensor, when we chain many layers together, and call the Model function API
{% highlight python %},
model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
{% endhighlight %}
in general, keras use directed acyclic graph(DAG) to represent model, and the DAG
have a name in keras called `Container`, `Model` class is derived from `Container`.

several points container have:
1. facility to save/load weights and the DAG architecture, convert to json etc.
2. mapping to tf loss and train.
3. input size is flexible

### Conclude
It's not complex to read keras source code, and the design is clear, concept behind it is very simple, but this make the powerful high level interface to build many kind of neural networks.