---
layout: post
title:  "Debug neural network code in Pytorch"
date:   2018-10-08 09:15:50 +0800
categories: debug pdb pytorch
---

I have used keras/tensorflow for one year, there are some pain points for me to use them. 
1. it can not set breakpoint  in the code. 
2. it can't watch the data during training/testing.

recently I found pytorch can address those problems perfectly.

I can use pdb to set breakpoint and check the data during running in GPU. for example, I will use a Transformer model and break at any point in it and check the data.

clone the example
```
git clone https://github.com/dongjk/Baidu-Traffic-speed-prediction.git
cd Baidu-Traffic-speed-prediction
```

run the program (need pre-processing data first)
```
python -m pdb train.py -model Transformer -dataset RoadDataSet3 -d_model=1 -epoch=16 -batch_size=64 -steps=1280000 -n_layers=2 -n_sample=62 -n_nb_sample=6 -n_max_seq=222 -d_inner=1024 -dropout=0.1
```

set Breakpoint
```
(Pdb++) b Layers.py:256
Breakpoint 1 at /root/jk/road/Layers.py:256
```

run to breakpoint
```
(Pdb++) c
[9] > /root/jk/road/Layers.py(256)Transformer()
-> def forward(self, x, mask=None):
(Pdb++) l
251     #         self.head3=nn.Linear(w*16,1)
252             self.head1=nn.Linear(n_max_seq*d_model,1)
253             self.head2=nn.Linear(n_max_seq*d_model,1)
254             self.head3=nn.Linear(n_max_seq*d_model,1)
255
256 B->     def forward(self, x, mask=None):
257             #x=self.fc(x)
258             b, len_q, d_model = x.size()
259             #b=temporal_pos_emb+x + spatial_pos_emb#add pos emb to input
260             if mask is not None:
261                 assert mask.dim() == 2
```

Now I can do whatever I want, check the data, check the model, etc, the whole process is straightforward and elegent


