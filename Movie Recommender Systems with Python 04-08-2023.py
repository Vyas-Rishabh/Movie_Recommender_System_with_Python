#!/usr/bin/env python
# coding: utf-8

# # ![movie-recommendation-systempptx-1-2048.JPG](attachment:movie-recommendation-systempptx-1-2048.JPG)

# Welcome to the code notebook for Recommender Systems with Python. In this lecture we will develop basic recommendation systems using Python and pandas.</br>
# </br>
# In this notebook, we will focus on providing a basic recommendation system by suggesting items that are most similar to a particular item, in this case, movies. Keep in mind, this is not a true robust recommendation system, to describe it more accurately,it just tells you what movies/items are most similar to your movie choice.</br>
# </br>
# There is no project for this topic, instead you have the option to work through the advanced lecture version of this notebook (totally optional!).
# 
# Let's get started!

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd


# ## Get the Data

# In[2]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df= pd.read_csv('u.data', names=column_names, sep='\t')


# In[3]:


df.head()


# Now let's get the movie titles:

# In[4]:


movie_titles = pd.read_csv('Movie_Id_Titles')
movie_titles.head()


# We can merge them together

# In[5]:


df = pd.merge(df,movie_titles, on='item_id')


# In[6]:


df.head()


# In[7]:


pd.merge(df,movie_titles, on='item_id').head(20)


# # EDA

# Let's Explore the data a bit and get a look at some of the best rated movies.

# ## Visualization Imports

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's create a ratings dataframe with average rating and number of rating:

# In[9]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[10]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[11]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[12]:


ratings.head()


# Now set the number of rating column:

# In[13]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[14]:


ratings.head()


# Now a few histogram

# In[15]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70);


# In[16]:


plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# In[17]:


sns.jointplot(x='rating', y='num of ratings', data=ratings,alpha=0.5)


# Okay! Now that we have a general idea of what the data look like, let's move on to creating a simple recommendation system:

# ## Recommending Similar Movies

# Now let's create a matrix that has the user ids on one access and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values, because most people have not seen most of the movies.

# In[18]:


moviemat = df.pivot_table(index='user_id', columns='title', values='rating')


# In[19]:


moviemat


# Most related movies:

# In[20]:


ratings.sort_values('num of ratings', ascending=False).head(10)


# Let's choose two movies: Starwars, a sci-fi movies. And Liar Liar, a comedy.

# In[21]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']


# In[22]:


starwars_user_ratings.head()


# We can then use corrwith() method to get correlations between two pandas series:

# In[23]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# In[24]:


similar_to_starwars


# In[25]:


corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])


# In[26]:


corr_starwars.dropna(inplace=True)
corr_starwars.head()


# Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie).

# In[27]:


corr_starwars.sort_values('Correlation', ascending=False).head(10)


# Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).

# In[28]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Now sort the values and notice how the titles make a lot more sense:

# In[29]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head(10)


# Now the same for the comedy Liar Liar:

# In[30]:


corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head(10)


# You can find this project on <a href="https://github.com/Vyas-Rishabh/Movie_Recommender_System_with_Python"><B>GitHub.</B></a>

# In[ ]:




