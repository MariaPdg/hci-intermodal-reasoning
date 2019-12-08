#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import scipy.misc 


# In[4]:


class Logger(object):
    
    def __init__(self, log_dir):
        
        def __init__(self, logdir):
            self.writer = tf.summary.FileWriter(logdir)

        def close(self):
            self.writer.close()

        def log_scalar(self, tag, value, global_step):
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=value)
            self.writer.add_summary(summary, global_step=global_step)
            self.writer.flush()


# In[ ]:




