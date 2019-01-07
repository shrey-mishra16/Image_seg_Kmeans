#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[35]:


img = cv2.imread("ron.jpeg")
im1=img

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[25]:


or_shape =img.shape


# In[6]:


plt.imshow(img)
plt.show()


# In[7]:


#Flattening The Image
all_pix = img.reshape((-1,3))


# In[8]:


all_pix.shape


# In[12]:


#For Reimaging To Dominant Color


from sklearn.cluster import KMeans


# In[15]:


dom_col = 4
km = KMeans(n_clusters=dom_col)
km.fit(all_pix)


# In[16]:


centers = km.cluster_centers_  #Centers For Dominant Colors


# In[17]:


centers = np.array(centers,dtype='uint8')


# In[18]:


print(centers)


# # Plotting dominant Colors

# In[19]:


i = 1

plt.figure(0,figsize=(8,2))


colors = []

for each_col in centers:
    plt.subplot(1,4,i)
    plt.axis("off")
    i+=1
    
    colors.append(each_col)
    
    #Color Swatch
    a = np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = each_col
    plt.imshow(a)
    
plt.show()


# # Segmenting Imge With 4 Dominant Colors

# In[20]:


c_img = np.reshape(img, (-1,3))


# In[21]:


c_img.shape


# In[24]:


for ix in range(c_img.shape[0]):
    c_img[ix] = centers[km.labels_[ix]]


# In[29]:


d_image = c_img.reshape((or_shape))


# In[36]:


plt.imshow(d_image)
plt.show()


# In[38]:


plt.imshow(im1)
plt.show()


# In[ ]:




