#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2019/9/11 18:45 
 @Author : ZHANG 
 @File : prepare_data.py 
 @Description:
"""

import json

'''''
####################
with open("data/annotation.json") as js:
    it = json.load(js)
    a = it['categories']

with open('data/zsd_seen.names', 'w') as f:
    for k in a:
        f.write(k['name'] + '\n')

###########################

with open("data/embeddings_GloVe.json") as js:
    it = json.load(js)
    a = it.keys()

with open('data/zsd.names', 'w') as f:
    for k in a:
        f.write(k + '\n')

#########################
with open("data/embeddings_GloVe.json") as js:
    it = json.load(js)

with open('data/zsd.emb', 'w') as f:
    with open('data/zsd.names', 'w') as f1:
        for k, v in it.items():
            f.write(str(v) + '\n')
            f1.write(k + '\n')

########################

with open("data/embeddings_GloVe.json") as js:
    emb = json.load(js)

with open("data/coco.names") as j:
    names = j.read().split('\n')

with open('data/coco.emb', 'w') as f:
    for nn in names:
        for k, v in emb.items():
            if k == nn:
                f.write(str(v) + '\n')
                

import os

a = open("data/val8000.txt", "w")
for path, subdirs, files in os.walk(r'/dlwsdata3/public/ZSD/ZJLAB_ZSD_2019/images/val'):
   for filename in files:
     f = os.path.join(path, filename)
     a.write(str(f) + os.linesep)
     
'''

import ast
a = []
import numpy as np
with open("data/coco.emb") as f:
    for line in f:
        l = ast.literal_eval(line)
        a.append(l)
