#!/usr/bin/env python
# coding: utf-8

# # For Multiple Dominant Color Value

# In[60]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans 


# In[61]:


n1=input("Min Number Of Dominant Colors  :  ")
n2=input("Mid Number Of Dominant Colors  :  ")
n3=input("Max Number Of Dominant Colors  :  ")

img = cv2.imread("irnman.jpeg")
im1=img

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[62]:


o_img = img.copy()
c_img3 = img.copy()
c_img5 = img.copy()
c_img7 = img.copy()
c_img3 = np.reshape(c_img3, (-1,3))
c_img5 = np.reshape(c_img5, (-1,3))
c_img7 = np.reshape(c_img7, (-1,3))


# In[63]:


kmeans3 = KMeans(n_clusters=int(n1),random_state=2)
kmeans5 = KMeans(n_clusters=int(n2),random_state=2)
kmeans7 = KMeans(n_clusters=int(n3),random_state=2)


# In[64]:


kmeans3.fit_predict(c_img3)
kmeans5.fit_predict(c_img5)
kmeans7.fit_predict(c_img7)


# In[65]:


centers3 = kmeans3.cluster_centers_  #Centers For 4 Dominant Colors
centers5 = kmeans5.cluster_centers_  #Centers For 6 Dominant Colors
centers7 = kmeans7.cluster_centers_  #Centers For 8 Dominant Colors



centers3 = np.array(centers3,dtype='uint8')
centers5 = np.array(centers5,dtype='uint8')
centers7 = np.array(centers7,dtype='uint8')


# # Plotting Dominant Colors

# In[66]:


centers = np.array([centers3,centers5,centers7])
for ix in range(3):
    i = 1
    plt.figure(0,figsize=(8,2))
    colors = []
    for each_col in centers[ix]:
        plt.subplot(1,centers[ix].shape[0],i)
        plt.axis("off")
        i+=1
        colors.append(each_col)
        #Color Swatch
        a = np.zeros((100,100,3),dtype='uint8')
        a[:,:,:] = each_col
        plt.imshow(a)

    plt.show()


# # Segmenting Image With Dominant Colors

# In[67]:


for ix in range(c_img3.shape[0]):
    c_img3[ix] = centers3[kmeans3.labels_[ix]]
    
for ix in range(c_img5.shape[0]):
    c_img5[ix] = centers5[kmeans5.labels_[ix]]
    
for ix in range(c_img7.shape[0]):
    c_img7[ix] = centers7[kmeans7.labels_[ix]]


# In[68]:


d_img3 = np.reshape(c_img3,(img.shape[0],img.shape[1],3))
d_img5 = np.reshape(c_img5,(img.shape[0],img.shape[1],3))
d_img7 = np.reshape(c_img7,(img.shape[0],img.shape[1],3))


# In[69]:


plt.figure(1,figsize=(20,20))


plt.subplot(2, 2, 1)
plt.axis("off")
plt.title("Original Image")
plt.imshow(im1)

plt.subplot(2,2,2)
plt.axis("off")
plt.title(n1+" Dominant Colors")
plt.imshow(d_img3)

plt.subplot(2,2,3)
plt.axis("off")
plt.title(n2+" Dominant Colors")
plt.imshow(d_img5)

plt.subplot(2,2,4)
plt.axis("off")
plt.title(n3+" Dominant Colors")
plt.imshow(d_img7)

plt.show()


# In[ ]:





# In[ ]:




