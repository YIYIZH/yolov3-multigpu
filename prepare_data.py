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

with open("data/visual.json.json") as js:
    emb = json.load(js)

with open("data/coco.vs") as j:
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
     


import ast
a = []
import numpy as np
with open("data/coco.emb") as f:
    for line in f:
        l = ast.literal_eval(line)
        a.append(l)
        


#########################
with open("data/embeddings_GloVe.json") as js:
    emb = json.load(js)

with open("data/zsd_seen.names") as j:
    names = j.read().split('\n')

a= open('data/zsd_unseen.emb', 'w')
with open('data/zsd_unseen.names', 'w') as f:
    for k, v in emb.items():
        if k not in names:
            f.write(str(k) + '\n')
            a.write(str(v) + '\n')

import csv

b= open('out/temp.txt', 'w')
with open('out/unseen.txt', 'r') as f:
    mm = 0
    for line in f:
        with open('out/onlyseenobj.txt', 'r') as j:
            for ll in j:
                fn = line.split(' ')
                jn = ll.split(' ')
                if fn[0] == jn[0]:
                    mm = 1
                    break
        if mm == 0:
            b.write(line)
        else:
            mm = 0

# cat cat.txt temp.txt > final_v1.txt

with open("data/visual.json") as js:
    emb = json.load(js)

with open("data/coco.names") as j:
    names = j.read().split('\n')

with open('data/coco.vs', 'w') as f:
    for nn in names:
        for k, v in emb.items():
            if k == nn:
                f.write(str(v) + '\n')
                      
'''''

a= open('out29/cat29.txt', 'w')
with open('out/onlyseenobj.txt', 'r') as f:
    lf = f.readlines()
    nf = len(lf)

with open('out29/unseen.txt', 'r') as j:
    lj = j.readlines()
    nj = len(lj)

q = 0
p = 0
while q < nf or p < nj:
    if q == nf:
        a.write(lj[p])
        p +=1
        continue
    if p == nj:
        a.write(lf[q])
        q +=1
        continue
    imgf = lf[q].split(' ')
    imgj = lj[p].split(' ')
    if imgf[0] < imgj[0]:
        a.write(lf[q])
        q += 1
        continue
    elif imgf[0] == imgj[0]:
        for z in range(len(imgf) - 1):
            a.write(str(imgf[z]) + ' ')
        for k in range(len(imgj) - 1):
            if k == 0:
                a.write(str(imgj[k + 1]))
            else:
                a.write(' ' + str(imgj[k + 1]))
        q +=1
        p +=1
        continue
    else:
        a.write(lj[p])
        p += 1
        continue