---
layout: post
title:  "To Categories methods"
date:   2018-03-14 10:25:50 +0800
categories: data_processing tips
---
{% include mathjax.html %}
In machine learning, we need map categories strings like 'dog', 'cat' to intger number.
in this post, I list some 'to categroies' tools
## sklearn 
sklearn have a pattern to do this:
1. get dataset
2. feed dataset to encoder use `fit()` to learn whole picture mapping
3. feed dataset to encoder use `transform()` to get mapped dataset

`LabelEncoder` will covert string to integer number but not one hot type

e.g, ['cat', 'dog'] map to [1, 2]
{% highlight python %},
>>> from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"]) 
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
{% endhighlight %}


use numpy to convert integer numbver to ont hot

{% highlight python %}
np.eye(n_class)[int_num_list]
{% endhighlight %}

from 0.20, `CategoricalEncoder` will directly covert string to one hot.

{% highlight python %}
>>> from sklearn.preprocessing import CategoricalEncoder
>>> enc = CategoricalEncoder(handle_unknown='ignore')
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
... 
CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
          encoding='onehot', handle_unknown='ignore')
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
array([[ 1.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.]])
>>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
array([['Male', 1],
       [None, 2]], dtype=object)
{% endhighlight %}

## keras
keras use `Tokenizer` to map strings to integer.
it have pattern similiar to sklearn
1. get dataset
2. feed dataset to tokenizer use `fit_on_texts()`.
3. feed dataset to tokenizer use `texts_to_sequences()`.

different with sklearn, tokenizer will not use `0` index.

e.g, ['cat', 'dog'] map to [1, 2]

`Tokenizer` will not convert to one hot format.

{% highlight python %}
encoder_tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
encoder_tokenizer.fit_on_texts(X)
encoder_seq = encoder_tokenizer.texts_to_sequences(X)
{% endhighlight %}

use `keras.utils.to_categorical` to convert integer sequence to one hot.

{% highlight python %}
to_categorical(encoder_seq[start:end], ENCODER_TOKEN_LENGTH + 1)
{% endhighlight %}

because this is no `0`, but category will contains `-`, so here need add 1 for 
the lenth.