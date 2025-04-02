#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#%%
# Load datasets
users = pd.read_csv('Users.csv')
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')

# # Get dataset info

# users.info()
# books.info()
# ratings.info()

# %%
# Get dataset info

users.info()
books.info()
ratings.info()

# %%

users

# %%
books
# %%
ratings
# %%
# Drop rows with duplicate book title
new_books = books.drop_duplicates('Book-Title')

# %%
new_books
# %%
print(len(books), len(new_books))
print(len(new_books) - len(books))

## - Negative sign says that the Duplicates have been remooved
# %%
# Merge ratings and new_books df

# before merging Columns Count 
print(f"Rating Column count = {(ratings.shape[1])} \n new_books Column count = {new_books.shape[1]}")
#%%

ratings_with_name = ratings.merge(new_books, on='ISBN')

#%%  
# Seepreviously there were 3 +8 = 11 columns but since we merged by ISBNN it gets added and reduced the no by one
ratings_with_name

#%%
# Drop non-relevant columns
ratings_with_name.drop(['ISBN', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis = 1, inplace = True)

# %%
ratings_with_name
# %%
# Merge new 'ratings_with_name' df with users df
users_ratings_matrix = ratings_with_name.merge(users, on='User-ID')

# Drop non-relevant columns
users_ratings_matrix.drop(['Location', 'Age'], axis = 1, inplace = True)

# Print the first few rows of the new dataframe
users_ratings_matrix.head()

# %%
# Check for null values
users_ratings_matrix.isna().sum()

# %%
# Drop null values
users_ratings_matrix.dropna(inplace = True)
print(users_ratings_matrix.isna().sum())

# %%
# Filter down 'users_ratings_matrix' on the basis of users who gave many book ratings
x = users_ratings_matrix.groupby('User-ID').count()['Book-Rating'] > 100
x

#%%
knowledgeable_users = x[x].index
knowledgeable_users
#%%
filtered_users_ratings = users_ratings_matrix[users_ratings_matrix['User-ID'].isin(knowledgeable_users)]

# Filter down 'users_ratings_matrix' on the basis of books with most ratings
y = filtered_users_ratings.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_users_ratings = filtered_users_ratings[filtered_users_ratings['Book-Title'].isin(famous_books)]

# %%
famous_books
# %%
filtered_users_ratings

#%% 
filtered_users_ratings.columns

# %%
# Pivot table creation
pivot_table = final_users_ratings.pivot_table(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating')
pivot_table
#%%
# Filling the NA values with '0'
pivot_table.fillna(0, inplace = True)
pivot_table.head()

# %%
# Standardize the pivot table
scaler = StandardScaler(with_mean=True, with_std=True)
pivot_table_normalized = scaler.fit_transform(pivot_table)

# Convert back to DataFrame
normalized_df = pd.DataFrame(
    pivot_table_normalized, 
    columns=pivot_table.columns, 
    index=pivot_table.index
)

# Heatmap visualization
sns.heatmap(normalized_df, annot=True, cmap='coolwarm', center=0)
plt.title('Standardized Pivot Table')
plt.show()

# %%
# Calculate the similarity matrix for all the books
similarity_score = cosine_similarity(pivot_table_normalized)
# similarity_score

#%% 
def recommend(book_name):
    
    # Returns the numerical index for the book_name
    index = np.where(pivot_table.index==book_name)[0][0]
    
    # Sorts the similarities for the book_name in descending order
    similar_books = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1], reverse=True)[1:6]
    
    # To return result in list format
    data = []
    
    for index,similarity in similar_books:
        item = []
        # Get the book details by index
        temp_df = new_books[new_books['Book-Title'] == pivot_table.index[index]]
        
        # Only add the title, author, and image-url to the result
        item.extend(temp_df['Book-Title'].values)
        item.extend(temp_df['Book-Author'].values)
        item.extend(temp_df['Image-URL-M'].values)
        
        data.append(item)
    return data

#%%

# Validating the model 
# Call the recommend method
recommend('1984',similarity_score)

#%%
#  Conclusion : Building a recommendation engine using collaborative filtering is a robust way to enhance personalization in services.
# %%
