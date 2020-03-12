#!/usr/bin/env python
# coding: utf-8

# # IKpy Quick-start #

# # Requirements

# First, you need to install IKPy (see [installations instructions](https://github.com/Phylliade/ikpy)).
# You also need a URDF file.  
# By default, we use the files provided in the [resources](https://github.com/Phylliade/ikpy/tree/master/resources) folder of the IKPy repo.

# Import the IKPy module : 

# In[1]:


import sys
sys.path.append("../src/")
import ikpy
import numpy as np
from ikpy import plot_utils
from IPython import get_ipython
import matplotlib.pyplot as plt
# The basic element of IKPy is the kinematic `Chain`.
# To create a chain from an URDF file : 

# In[2]:


my_chain = ikpy.chain.Chain.from_urdf_file("../resources/poppy_ergo.URDF")


# Note : as mentioned before, here we use a file in the resource folder.

# # Inverse kinematics

# In Inverse Kinematics, you want your kinematic chain to reach a 3D position in space.

# To have a more general representation of position, IKPy works with homogeneous coordinates. It is a 4x4 matrix storing both position and orientation.
# Prepare your desired position as a 4x4 matrix. Here we only consider position, not orientation of the chain.

# In[25]:


target_vector = [ 0.05, 0.05, 0.15]
#target_frame = np.eye(4)
#target_frame = np.ones((4,4))
#target_frame = np.zeros((4,4))
target_frame = np.random.random((4,4))
target_frame[:3, 3] = target_vector


# In[27]:


print(target_frame)
six_axis = my_chain.inverse_kinematics(target_frame)
print("The angles of each joints are :\n", six_axis)
print(my_chain)


# You can check that the Inverse Kinematics is correct by comparing with the original position vector : 

# In[28]:


real_frame = my_chain.forward_kinematics(six_axis)
print("Computed position vector :\n %s,\n original position vector :\n %s" % (real_frame, target_frame))
target_vector_ag = real_frame[:3,3]
six_axis_ag = my_chain.inverse_kinematics(real_frame)
print("The angles of each joints ag are :\n", six_axis_ag)


# # Plotting
# And finally plot the result : 

# (If the code below doesn't work, comment the `%maplotlib notebook` line, and uncomment the `%matplotlib inline` line)

# In[14]:


# If there is a matplotlib error, uncomment the next line, and comment the line below it.
#%matplotlib inline
#import matplotlib.pyplot as plt
ax = plot_utils.init_3d_figure()
my_chain.plot(six_axis, ax, target=target_vector)
plt.xlim(-0.1, 0.1)
plt.ylim(-0.1, 0.1)


# You're done! Go to the [tutorials](https://github.com/Phylliade/ikpy/wiki) to understand the general concepts of the library.

# In[13]:


# If there is a matplotlib error, uncomment the next line, and comment the line below it.
#%matplotlib inline
#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
ax = plot_utils.init_3d_figure()
my_chain.plot(six_axis_ag, ax, target=target_vector_ag)
plt.xlim(-0.1, 0.1)
plt.ylim(-0.1, 0.1)

plt.show()
