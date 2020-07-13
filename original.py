#!/usr/bin/env python
# coding: utf-8

# ## Step05_kNearestNeighbor
# 
# k最近傍法で分類する
# 混同行列と各クラスの認識率を表示する

# In[21]:


from skimage import io
import numpy as np


# In[22]:


TrainingSampleNum = 2000 # 学習サンプル総数
TestSampleNum = 10000 # テストサンプル総数
ClassNum = 10 # クラス数（今回は10）
ImageSize = 28 # 画像サイズ（今回は縦横ともに28）
TrainingDataFile = './Images/TrainingSamples/{0:1d}-{1:04d}.png'
TestDataFile = './Images/TestSamples/{0:1d}-{1:04d}.png'
OutFile = './Images/OutSamples/gray_{0:1d}-{1:04d}.png'


# In[23]:


# LoadTemplates
def LoadTemplates():
    labels = np.zeros(TrainingSampleNum, dtype=np.uint8)
    templates = np.zeros((TrainingSampleNum,ImageSize,ImageSize), dtype=np.int16)

    i = 0
    for label in range(0, ClassNum):
        for sample in range(0, TrainingSampleNum // ClassNum):
            filename = TrainingDataFile.format(label, sample)
            templates[i,:,:] = io.imread(filename).astype(np.int16)
            labels[i]=label
            i += 1

    return templates, labels


# In[24]:


# CalcDistance
# 今回は L1 距離を算出
def CalcDistance(target, template):
    buf = np.fabs(target - template)
    return np.sum(buf)


# In[25]:


# ReturnMatchLabel
def ReturnMatchLabel(target, templates, labels, K):
    min_dis = np.full(K, float("Inf"))
    k_labels = np.full(K, -1)
    
    for i in range(0, TrainingSampleNum):
        dis = CalcDistance(target, templates[i,:,:])
        label = labels[i]
        for k in range(0, K):
            if min_dis[k] > dis:
                min_dis[k], dis = dis, min_dis[k]
                k_labels[k], label = label, k_labels[k]

    class_count = np.zeros(ClassNum)
    for k in range(0, K):
        class_count[k_labels[k]] += 1
        
    match_label = 0;
    max_count = 0;
    for i in range(0, ClassNum):
        if max_count < class_count[i]:
            max_count = class_count[i]
            match_label = i
    
    return match_label


# In[26]:


# kNearestNeighbor
def kNearestNeighbor(templates, labels, k):

    results = np.zeros((ClassNum, ClassNum))
    each_class_num = TestSampleNum // ClassNum
    
    for label in range(0, ClassNum):
        for sample in range(0, each_class_num):
            filename = TestDataFile.format(label, sample)
            t_img = io.imread(filename).astype(np.int16)
            match_label = ReturnMatchLabel(t_img, templates, labels, k)
            results[label, match_label] += 1
        print("{0:1d}: {1:.4f}".format(label, results[label, label] / each_class_num))

    print("= Confusion matrix ===========")
    for t_label in range(0, ClassNum):
        for m_label in range(0, ClassNum):
            print("{:04g}, ".format(results[t_label, m_label]), end="")
        print("")

    print("= Total Recognition accuracy ===========")
    total_correct_num = 0
    for t_label in range(0, ClassNum):
        total_correct_num += results[t_label, t_label]
    print ("TOTAL: {0:.4f}".format(total_correct_num / TestSampleNum))


# In[27]:


# main ルーチン
templates, labels = LoadTemplates()
k = 3
kNearestNeighbor(templates, labels, k)


# In[ ]:




