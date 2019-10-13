#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#----------------Reading Dataset----MOVIES
movies=pd.read_csv(r'E:\ACDS\Sem_2\ML\lab\datasets\movie_recommendation\movies.csv',sep='\t',encoding='latin-1',usecols=['movie_id','title','genres'])
movies


# In[3]:


#----------------Reading Dataset----USERS
users=pd.read_csv(r'E:\ACDS\Sem_2\ML\lab\datasets\movie_recommendation\users.csv',sep='\t',encoding='latin-1',
                  usecols=['user_id','gender','zipcode','age_desc','occ_desc'])
users


# In[4]:


#----------------Reading Dataset----RATINGS
ratings=pd.read_csv(r'E:\ACDS\Sem_2\ML\lab\datasets\movie_recommendation\ratings.csv',sep='\t',encoding='latin-1',
                  usecols=['user_id','movie_id','rating','timestamp'])
ratings


# In[5]:


#Count 

#Number of unique users:
n_users=ratings.user_id.unique().shape[0]
n_users


# In[6]:


#Number of unique movies:
n_movies=ratings.movie_id.unique().shape[0]
n_movies


# In[7]:


#------Converting it to matrix form

#from ratings datset  check for NAN values so  fill it with 0

Ratings=ratings.pivot(index='user_id',columns='movie_id',values='rating').fillna(0)
Ratings.head()


# In[ ]:





# In[8]:


#Normalize by each users mean ad convert it from a dataframe to  a numpy array:

R=Ratings.as_matrix()
users_ratings_mean=np.mean(R,axis=1)
Ratings_demeaned=R-users_ratings_mean.reshape(-1,1)
Ratings_demeaned


# In[9]:


''' **Sparsity Matrix**

   11 22 0  0  0  0
   0  33 0  0  0  0
   0  0  66 0  0  0
   0  11 55 77 0  0
   0  0  0  44 0  0
   0  0  0  0  11 0  
    
--in matrix maximum values are 0
--we get the values diagonaly and they are distinct
--It is used in Numerical analysis
--A sparse matrix is  a matrix in which most the values are zero '''  


# In[10]:


#-------------Sparsity:

sparsity=round(1.0-len(ratings)/float(n_users*n_movies),3)
print("the sparsity level of MovieLens1M dataset is "+str(sparsity*100)+'%')


# In[11]:


#---------------SVD:
#Calculte ur svd --we get-->u,sigma,vt
#k-->number of singualr values

from scipy.sparse.linalg import svds
U,sigma,Vt=svds(Ratings_demeaned,k=50)


# In[12]:


print('Sigma:\n ',sigma)

print('U:\n ',U)

print('Vt:\n',Vt)


# In[13]:


#To view it diagonally:

sigma=np.diag(sigma)
sigma


# In[14]:


#---------------Predictions: 
#calculated : u*vt*sigma

all_user_predicted_ratings=np.dot(np.dot(U,sigma),Vt) + users_ratings_mean.reshape(-1,1)
all_user_predicted_ratings


# In[15]:


#converted predicted ratings into df so that our data will be in rows and columns:

preds=pd.DataFrame(all_user_predicted_ratings,columns=Ratings.columns)
preds.head()


# In[16]:


#Defining a Function
#user_data-->user_id-->id has recommendation of movie ------>change
def recommend_movies(prediction,userID,movies,original_ratings,num_recommendations):
    #get and sort the user prediction:
    user_row_number=userID - 1       #User Id starts with 1 , not 0
    #descending order arrangements of user predictions:
    sorted_user_predictions=preds.iloc[user_row_number].sort_values(ascending=False)   #User id starts at 1
    
    #Get the users data and merge in the movie info:
    #-->doing so beacuse movie will be recommended to user_id
    user_data=original_ratings[original_ratings.user_id==(userID)]
    user_full=(user_data.merge(movies,how='left',left_on='movie_id',right_on='movie_id').sort_values(['rating'],ascending=False))
    
    print('User{0} has already rated {1} movies .'.format(userID,user_full.shape[0]))
    
    print('Recommending highest {0} predicted ratings movies not already rated '.format(num_recommendations))
    
    #recommend the highest predicted rating movies that the user hasnt seen yet.
    
    recommendations=(movies[~movies['movie_id'].isin(user_full['movie_id'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(),
                                                                                 how='left',
                                                                                  left_on='movie_id',
                                                                                 right_on='movie_id').
                    rename(columns={user_row_number:'Predictions'}).sort_values('Predictions',ascending=False).iloc[:num_recommendations,
                                                                                                                   :-1])
                
    return user_full,recommendations                
                    
    


# In[17]:


already_rated,predictions=recommend_movies(preds,1310,movies,ratings,20)


# In[18]:


#Top 20 movies that user 1310 has rated:

already_rated.head(20)


# In[19]:


#-------Recommendations-/ Prediction:

#top 20movie suggestion that user 1310 hopefully will enjoy

predictions


# In[ ]:




