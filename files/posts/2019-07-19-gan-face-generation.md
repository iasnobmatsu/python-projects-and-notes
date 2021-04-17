---
layout: post
title:  "Generative Adversarial Net Face Generation"
date: 2019-7-19
excerpt: "implementing GAN using tenserflow and generating celebrity faces on small scale with relatively low resolutions and small image sizes in python"
---

This is my kaggle kernal from [here](https://www.kaggle.com/iasnobmatsu/gan-face-generation).

### WorkFlow
- Data Exploration
- Data Cleaning
- GAN variables setup
- GAN network setup
- GAN Cost and Optimizer
- GAN iteration
- View outputs


```python
from PIL import Image
import os
root="../input/img_align_celeba/img_align_celeba/"
allim=os.listdir("../input/img_align_celeba/img_align_celeba")
allim[:10]
print(len(allim))
#get a small sample
allim=allim[:20000]
print(len(allim))
```

    202599
    20000


#### Data Exploration


```python
#visualize data
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
for i in range (30):
    impath=root+allim[i]
    im=Image.open(impath)
    plt.subplot(3,10,i+1)
    plt.imshow(im)
```


![png]({{site.baseurl}}/images/gan-face-generation_files/gan-face-generation_3_0.png)


#### Data Cleaning 


```python
# transfer image to numpy array, resize 3d to 1d
import numpy as np
def getImage(id,w=30,h=36):
    path=root+id
    im=Image.open(path)
    im=im.resize([w,h],Image.NEAREST)
    im=np.array(im)
    im=im.reshape(w*h*3)
    return im


for i in range(len(allim)):
    allim[i]=getImage(allim[i])


print(allim[3].shape)
print(len(allim))
    

```

    (3240,)
    20000



```python
#visualize resized images
plt.figure(figsize=(12,6))
for i in range (30):
    plt.subplot(3,10,i+1)
    plt.imshow(allim[i].reshape(36,30,3))
```


![png]({{site.baseurl}}/images/gan-face-generation_files/gan-face-generation_6_0.png)



```python
#normalize data for tanh(GAN generation function)
allim=np.array(allim)
allim=allim/255*2-1
allim.shape
```




    (20000, 3240)



#### GAN variables setup


```python
#input real image and noise
import tensorflow as tf
def inputs(dim_real,dim_noise):
    input_reals=tf.placeholder(tf.float32, [None, dim_real], name='input_reals')
    input_noises=tf.placeholder(tf.float32, [None, dim_noise], name='input_noises')
    return input_reals, input_noises

```


```python
#generator
def generator(noises,nn_units,out_dimension,alpha=0.01,reuse=False):
    
    with tf.variable_scope("generator", reuse=reuse):
        hidden1=tf.layers.dense(input_noises,nn_units)
        #leaky relu
        hidden1=tf.maximum(alpha*hidden1, hidden1)
        hidden1=tf.layers.dropout(hidden1,rate=0.2)

        logits=tf.layers.dense(hidden1, out_dimension)
        outputs=tf.tanh(logits)

        return logits,outputs
```


```python
#discriminator
def discriminator(image,nn_units,alpha=0.01,reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        hidden1=tf.layers.dense(image,nn_units)
        #leaky relu
        hidden1=tf.maximum(alpha*hidden1, hidden1)

        logits=tf.layers.dense(hidden1, 1)
        outputs=tf.sigmoid(logits)

        return logits,outputs
```


```python
dim_real=allim[0].shape[0]
print(dim_real)
dim_noise=100
gen_units=128
dis_units=128
LR=0.001
alpha=0.01
```

    3240


#### GAN network setup


```python

tf.reset_default_graph()
input_reals, input_noises=inputs(dim_real,dim_noise)
gen_logits, gen_outputs=generator(input_noises,gen_units, dim_real)
dis_real_logits, dis_real_outputs=discriminator(input_reals,dis_units)
dis_fake_logits, dis_fake_outputs=discriminator(gen_outputs,dis_units, reuse=True)

```


```python
dis_real_cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logits,
                                                                     labels=tf.ones_like(dis_real_logits)))
dis_fake_cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits,
                                                                     labels=tf.zeros_like(dis_fake_logits)))
dis_total_cost=tf.add(dis_real_cost, dis_fake_cost)

gen_cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits,
                                                                labels=tf.ones_like(dis_fake_logits)))
```


```python
train_vars=tf.trainable_variables()
gen_vars=[var for var in train_vars if var.name.startswith('generator')]
dis_vars=[var for var in train_vars if var.name.startswith('discriminator')]
gen_optimizer=tf.train.AdamOptimizer(LR).minimize(gen_cost,var_list=gen_vars)
dis_optimizer=tf.train.AdamOptimizer(LR).minimize(dis_total_cost,var_list=dis_vars)
```

#### Gan iteration


```python
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

batch_size=64
count=0
g_cost=0
d_cost=0
samples=[]
for i in range(40000):
    startindex=(i*batch_size)%(len(allim)-batch_size)
    endindex=startindex+batch_size
    batch_real=allim[startindex:endindex] #shape is (64,3240)
    batch_noise=np.random.uniform(-1,1,size=(batch_size, dim_noise))
    sess.run(dis_optimizer,feed_dict={input_reals:batch_real,input_noises:batch_noise})
    sess.run(gen_optimizer, feed_dict={input_noises:batch_noise})
    g_cost+=(sess.run(gen_cost, feed_dict={input_noises:batch_noise}))
    d_cost+=(sess.run(dis_total_cost, feed_dict={input_reals:batch_real,input_noises:batch_noise}))
    
    if (i+1)%1000==0:
        count+=1
        print("ITER:",count,"| GEN COST:",g_cost/(1000),"DIS COST:",d_cost/(1000))
        g_cost=0
        d_cost=0
        gen_samples=sess.run(generator(input_noises,gen_units,dim_real, reuse=True),
                            feed_dict={input_noises:batch_noise})
        samples.append(gen_samples)
```

    ITER: 1 | GEN COST: 3.443799765303731 DIS COST: 0.10586929506622254
    ITER: 2 | GEN COST: 5.182610770463944 DIS COST: 0.010914721666471451
    ITER: 3 | GEN COST: 6.398988176584243 DIS COST: 0.04562442097370513
    ITER: 4 | GEN COST: 7.444714887857437 DIS COST: 0.20618675654230173
    ITER: 5 | GEN COST: 6.66671141242981 DIS COST: 0.3148060571386013
    ITER: 6 | GEN COST: 5.624609602689743 DIS COST: 0.18997684904746712
    ITER: 7 | GEN COST: 5.572898686170578 DIS COST: 0.25047424796409906
    ITER: 8 | GEN COST: 5.187875975847244 DIS COST: 0.3159035860262811
    ITER: 9 | GEN COST: 4.400113422632217 DIS COST: 0.35487657806277273
    ITER: 10 | GEN COST: 3.913784235715866 DIS COST: 0.382004903152585
    ITER: 11 | GEN COST: 3.8784089381694793 DIS COST: 0.38089552431181073
    ITER: 12 | GEN COST: 3.660636285662651 DIS COST: 0.42637104383856056
    ITER: 13 | GEN COST: 3.3267564650774 DIS COST: 0.4765779176205397
    ITER: 14 | GEN COST: 3.1747682107686996 DIS COST: 0.5021283072084188
    ITER: 15 | GEN COST: 3.11033195745945 DIS COST: 0.5258731587827206
    ITER: 16 | GEN COST: 3.1125116552114487 DIS COST: 0.6031663199961186
    ITER: 17 | GEN COST: 2.9016200420856477 DIS COST: 0.6161733454912901
    ITER: 18 | GEN COST: 2.7283099282979966 DIS COST: 0.6514441814124584
    ITER: 19 | GEN COST: 2.6640971239805222 DIS COST: 0.6861120727360248
    ITER: 20 | GEN COST: 2.473084646344185 DIS COST: 0.7098466829061508
    ITER: 21 | GEN COST: 2.386759070634842 DIS COST: 0.7452217847704887
    ITER: 22 | GEN COST: 2.225544359087944 DIS COST: 0.8203532241880894
    ITER: 23 | GEN COST: 2.1031593751907347 DIS COST: 0.8423090198934078
    ITER: 24 | GEN COST: 1.9830808180570603 DIS COST: 0.8945590399205685
    ITER: 25 | GEN COST: 1.985911282658577 DIS COST: 0.9275676882266999
    ITER: 26 | GEN COST: 1.8460248234272003 DIS COST: 0.9972911101579666
    ITER: 27 | GEN COST: 1.7817888078689574 DIS COST: 1.0085415793061256
    ITER: 28 | GEN COST: 1.7201407709121703 DIS COST: 1.0303745971918106
    ITER: 29 | GEN COST: 1.7330161080360413 DIS COST: 1.037250403523445
    ITER: 30 | GEN COST: 1.7797714570760728 DIS COST: 1.0452938777804375
    ITER: 31 | GEN COST: 1.6592338242530822 DIS COST: 1.1238225837945939
    ITER: 32 | GEN COST: 1.6117033113241195 DIS COST: 1.0836714025735854
    ITER: 33 | GEN COST: 1.6451644033193589 DIS COST: 1.0503634859919548
    ITER: 34 | GEN COST: 1.5288553164601326 DIS COST: 1.1883836540579795
    ITER: 35 | GEN COST: 1.4300855733156204 DIS COST: 1.1585196332931518
    ITER: 36 | GEN COST: 1.496381982088089 DIS COST: 1.1572670236229896
    ITER: 37 | GEN COST: 1.4841150160431862 DIS COST: 1.163256119966507
    ITER: 38 | GEN COST: 1.468531046807766 DIS COST: 1.1573680483698845
    ITER: 39 | GEN COST: 1.448049015581608 DIS COST: 1.1676044368743896
    ITER: 40 | GEN COST: 1.4180886632204055 DIS COST: 1.1712574068307877



```python
sess.close()
```

#### View output


```python
print(len(samples))
print(len(samples[1]))
print((samples[0][0].shape))
```

    40
    2
    (64, 3240)



```python
#generated logits
samples[0][0][0]
```




    array([ 0.08103737,  0.09352005,  0.17542179, ..., -0.02888909,
            0.00361089, -0.08807059], dtype=float32)




```python
#generated outputs
samples[0][1][0]
```




    array([ 0.08086044,  0.09324836,  0.17364426, ..., -0.02888106,
            0.00361088, -0.08784359], dtype=float32)




```python
plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    img=((samples[0][1][i]+1)*255/2).astype(np.uint8)
    plt.imshow(img.reshape(36,30,3))
```


![png]({{site.baseurl}}/images/gan-face-generation_files/gan-face-generation_24_0.png)



```python
plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    img=((samples[9][1][i]+1)*255/2).astype(np.uint8)
    plt.imshow(img.reshape(36,30,3))
```


![png]({{site.baseurl}}/images/gan-face-generation_files/gan-face-generation_25_0.png)



```python
plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    img=((samples[39][1][i]+1)*255/2).astype(np.uint8)
    plt.imshow(img.reshape(36,30,3))
```


![png]({{site.baseurl}}/images/gan-face-generation_files/gan-face-generation_26_0.png)



```python
c=0
for i in range(64):
    path='img'+str(c)+'.jpg'
    img=((samples[39][1][i]+1)*255/2).astype(np.uint8)
    img=img.reshape(36,30,3)
    plt.imsave(path,img)
    c+=1
```
