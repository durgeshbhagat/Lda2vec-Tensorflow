#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from lda2vec.nlppipe import Preprocessor


# In[2]:


# Data directory
data_dir ="data"
# Where to save preprocessed data
clean_data_dir = "data/clean_data"
# Name of input file. Should be inside of data_dir
input_file = "20_newsgroups.txt"
# Should we load pretrained embeddings from file
load_embeds = True


# In[3]:


import os
# Read in data file
data_file_total = os.path.join(data_dir, input_file)
df = pd.read_csv(data_file_total, sep="\t")


# In[6]:


print(df.columns)
print(df.iloc[4,0])


# In[7]:


# Initialize a preprocessor
P = Preprocessor(df, "texts", token_type="lower", max_features=10000,
                 maxlen=10000, min_count=30, nlp="en_core_web_lg")


# In[8]:


# Run the preprocessing on your dataframe
P.preprocess()


# In[ ]:



# Load embeddings from file if we choose to do so
if load_embeds:
    # Load embedding matrix from file path - change path to where you saved them
    embedding_matrix = P.load_glove("glove.6B/glove.6B.300d.txt")
else:
    embedding_matrix = None


# In[ ]:


# Save data to data_dir
P.save_data(clean_data_dir, embedding_matrix=embedding_matrix)


# In[ ]:




