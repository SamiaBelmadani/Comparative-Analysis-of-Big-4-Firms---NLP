# The code is separated in 3 parts excluding the dataset creation. 
# Because the textual analysis preprocessing is different for all the parts, we load the initial dataset before every part.

#################################################################################################################################
### PART 1: Dataset Creation/Preprocessing
#################################################################################################################################

# Load
import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\olivi\\OneDrive\\Documents\\McGill MMA\\Winter 2024\\Text Analytics (INSY 669)\\Final project\\glassdoor_reviews.csv")

# see the names of all the firms
distinct_firms_df = df['firm'].drop_duplicates().to_frame()

# List of firms to include in the new dataset
firms_to_include = ['Accenture', 'AlixPartners', 'Bain-and-Company', 'BDO', 'CGI', 'Deloitte', 'EY', 'KPMG', 'McKinsey-and-Company', 'PwC', 'Grant-Thornton', 'RSM', 'Egon-Zehnder', 'Grant-Thornton-UK-LLP', 'JLL', 'McKinsey-and-Company', 'Oliver-Wyman', 'Serco-Group']

# Filter the original DataFrame based on the specified firms
df_consulting = df[df['firm'].isin(firms_to_include)]

# Replace empty or whitespace-only values in the "Job_Title" column with NaN
df_consulting['outlook'] = df_consulting['outlook'].replace('', np.nan).replace(' ', np.nan)

# Count the number of NaN values in the "Job_Title" column
nan_count = df_consulting['outlook'].isnull().sum()

print(f"Number of NaN values in the 'outlook' column after replacement: {nan_count}")


#12737 --> Missing values for Job Titles
# 0 --> Missing values for Current (employee) 
#544 --> Missing Values for Headline
#0 --> Missing values for Pros
#2 --> Missing values for Cons 
#45802 --> Missing values for location 
#0 --> Missing values for Overall Rating 
#24943 --> Missing values for Work_Life_Balance 
#32437 --> Missing values for Culture Values Rating 
#105578---> Missing values for Diversity Inclusion
#24570 --> Missing values for Career Opportunity 
#24857 --> Missing values for Company Benefits
#25678 --> Missing values for Senior Management Approval?
#0 --> Missing values for recommending 
#0 --> Missing values for CEO Approval 
#0 --> Missing values for Outlook 

df_consulting = df_consulting.dropna()


#RECOMMEND, CEO_APPROVAL & OUTLOOK 
#v - Positive, r - Mild, x - Negative, o - No opinion
encoding_dict = {'v': 1, 'r': 2, 'x': 3, 'o': 4}
df_consulting['recommend'] = df_consulting['recommend'].map(encoding_dict)
df_consulting['ceo_approv'] = df_consulting['ceo_approv'].map(encoding_dict)
df_consulting['outlook'] = df_consulting['outlook'].map(encoding_dict)

# Remove rows with value 4 in the specified columns
columns_to_check = ['ceo_approv', 'recommend', 'outlook']
for column in columns_to_check:
    df_consulting = df_consulting[df_consulting[column] != 4]

# Convert the 'date' column to datetime format
df_consulting['date_review'] = pd.to_datetime(df_consulting['date_review'])

# Get distinct values in the 'current' column
distinct_values = df_consulting['current'].unique()

print("Distinct values in the 'current' column:")
print(distinct_values)

# Create a new column 'current_employee' with 1 if 'Current Employee' is present, 0 otherwise
df_consulting['current_employee'] = df_consulting['current'].apply(lambda x: 1 if 'Current Employee' in x else 0)

# Extract number of years from the 'current' column using regular expressions
df_consulting['years_in_company'] = df_consulting['current'].str.extract(r'(\d+) year')

# Count NA values in the 'years_in_company' column
na_count = df_consulting['years_in_company'].isna().sum()

print(f"Number of NA values in the 'years_in_company' column: {na_count}")

# Drop rows with 'nan' values in the 'years_in_company' column
######### maybe not drop here because we get rid of 4123 rows #########
df_consulting = df_consulting.dropna(subset=['years_in_company'])

# Get distinct values in the 'job_titles' column
distinct_values = df_consulting['job_title'].unique()
print("Distinct values in the 'job_title' column:")
print(distinct_values)

# Count distinct values in the 'job_titles' column
distinct_counts = df_consulting['job_title'].value_counts()
print(distinct_counts)

# Count distinct values in the 'job_titles' column
distinct_counts = df_consulting['job_title'].value_counts().reset_index()
distinct_counts.columns = ['job_title', 'count']

# Save the new dataframe with only consulting firms
file_path = "C:\\Users\\olivi\\OneDrive\\Documents\\McGill MMA\\Winter 2024\\Text Analytics (INSY 669)\\Final project\\df_consulting.csv"
df_consulting.to_csv(file_path, index=False)  # Set index=False to exclude index column from CSV


### Average ratings
# Load data
import pandas as pd
df = pd.read_csv("C:\\Users\\olivi\\OneDrive\\Documents\\McGill MMA\\Winter 2024\\Text Analytics (INSY 669)\\Final project\\df_consulting.csv")

# Get average ratings from consulting in general
columns_to_average = ['overall_rating', 'work_life_balance', 'culture_values', 'diversity_inclusion', 'career_opp', 'comp_benefits', 'senior_mgmt']
average_ratings = df_consulting[columns_to_average].mean()
average_df = pd.DataFrame(average_ratings, columns=['average_rating'])

# Get average ratings from Big 4 firms
selected_firms = ['Deloitte', 'KPMG', 'EY', 'PwC']
df_big4 = df_consulting[df_consulting['firm'].isin(selected_firms)].copy()
# save dataset for future use
file_path = "C:\\Users\\olivi\\OneDrive\\Documents\\McGill MMA\\Winter 2024\\Text Analytics (INSY 669)\\Final project\\df_big4.csv"
df_big4.to_csv(file_path, index=False)  # Set index=False to exclude index column from CSV

columns_to_average = ['overall_rating', 'work_life_balance', 'culture_values', 'diversity_inclusion', 'career_opp', 'comp_benefits', 'senior_mgmt']
average_ratings = df_big4[columns_to_average].mean()
average_df = pd.DataFrame(average_ratings, columns=['average_rating'])

# Get average ratings from Big 4 firms individually
average_ratings = df_big4.groupby('firm')[columns_to_average].mean()
average_ratings = average_ratings.T



#################################################################################################################################
### PART 2: Sentiment Analysis
#################################################################################################################################

# Load data
import pandas as pd
df_big4 = pd.read_csv("C:/Users/julie/OneDrive/Documents/MMA/Winter 2024/Text Analytics/df_big4.csv")

####### Preprocessing ########

df_big4['pros_and_cons'] = df_big4['pros'] + " | " + df_big4['cons']
df_big4.drop(columns=['pros', 'cons'], inplace=True)

print(df_big4.columns)
# Remove duplicates
df_big4 = df_big4.drop_duplicates()

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import re
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nlp = spacy.load('en_core_web_sm')

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Define a function to clean the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove newline characters
    text = re.sub(r'\n', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'@\w+|#', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # lemmatization
    text = ' '.join([token.lemma_ for token in nlp(text)])
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


##### Pros and cons ######
# Apply the cleaning function to your text data column
df_big4['clean_text'] = df_big4['pros_and_cons'].apply(clean_text)

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis to the cleaned text
df_big4['sentiment'] = df_big4['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Aggregate the sentiment scores by company
sentiment_per_company = df_big4.groupby('firm')['sentiment'].mean()

# Display the sentiment scores per company
print(sentiment_per_company)


####### Headline only ########
# Apply the cleaning function to the headline column
df_big4['clean_headline'] = df_big4['headline'].apply(clean_text)

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis to the cleaned headline text
df_big4['headline_sentiment'] = df_big4['clean_headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Aggregate the sentiment scores by company from the headline column
sentiment_per_company_from_headline = df_big4.groupby('firm')['headline_sentiment'].mean()

# Display the sentiment scores per company based on headlines
print(sentiment_per_company_from_headline)


###### Headline Sentiment impact on overall rating #########
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Define the features to be scaled
features = ['headline_sentiment']

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale the features
df_big4[features] = scaler.fit_transform(df_big4[features])

# Continue with your regression analysis
X = df_big4[features]
y = df_big4['overall_rating']
X = sm.add_constant(X)  # adding a constant
model = sm.OLS(y, X).fit()
print(model.summary())


###### Scores impact on headline sentiment #########
import statsmodels.api as sm

# Define the independent variables (the ratings)
X = df_big4[['work_life_balance', 'culture_values', 'diversity_inclusion', 'career_opp', 'comp_benefits', 'senior_mgmt']]  # Add or remove columns based on your dataset

# Define the dependent variable (headline sentiment)
y = df_big4['headline_sentiment']

# Add a constant to the model 
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(model.summary())


####### Scores Impact on Overall Score ###########
import statsmodels.api as sm

# Define the independent variables (the ratings)
X = df_big4[['work_life_balance', 'culture_values', 'diversity_inclusion', 'career_opp', 'comp_benefits', 'senior_mgmt']]  # Add or remove columns based on your dataset

# Define the dependent variable (overall sentiment)
y = df_big4['overall_rating']

# Add a constant to the model
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(model.summary())



#################################################################################################################################
### PART 3.1: Dominant Topics in Reviews - Pros
#################################################################################################################################

#Import dataset 
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import spacy

df = pd.read_csv('/Users/samiabelmadani/Downloads/df_big4.csv')

##########################################
####For all Firms#########################
##########################################

####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')


def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|great|good|lot|company|work', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove text within parentheses
    x = re.sub(r'\([^)]*\)', '', x)
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
    return x

# Example usage
df['clean_pros'] = df['pros'].apply(clean_text)

#Remove empty rows
df = df[df['clean_pros'].str.strip() != '']

#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(df["clean_pros"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
df["Topic"] = topic_results.argmax(axis=1)

df.head()

###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)


from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)

####=====================DELOITTE###############

# Filter the DataFrame to include only rows where the "firm" column equals "DELOITTE"
deloitte_df = df[df['firm'] == 'Deloitte']

# Now deloitte_df contains only rows where the "firm" column is equal to "DELOITTE"
# You can use deloitte_df for further analysis or save it to a new file if needed

####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|great|good|lot|company|consulting|nice|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
deloitte_df['clean_pros'] = deloitte_df['pros'].apply(clean_text)

#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(deloitte_df["clean_pros"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
deloitte_df["Topic"] = topic_results.argmax(axis=1)

deloitte_df.head()


###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)

###==============================PWC===========================================
# Filter the DataFrame to include only rows where the "firm" column equals "PWC"
pwc_df = df[df['firm'] == 'PwC']


####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|great|good|lot|company|consulting|nice|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
pwc_df['clean_pros'] = pwc_df['pros'].apply(clean_text)

#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(pwc_df["clean_pros"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
pwc_df["Topic"] = topic_results.argmax(axis=1)

pwc_df.head()


###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)

###=============================EY============================================
# Filter the DataFrame to include only rows where the "firm" column equals "EY"
ey_df = df[df['firm'] == 'EY']


####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|great|good|lot|company|consulting|nice|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
ey_df['clean_pros'] = ey_df['pros'].apply(clean_text)

#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(ey_df["clean_pros"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
ey_df["Topic"] = topic_results.argmax(axis=1)

ey_df.head()


###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)

###==============================KPMG===========================================
# Filter the DataFrame to include only rows where the "firm" column equals "EY"
kpmg_df = df[df['firm'] == 'KPMG']


####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|great|good|lot|company|consulting|nice|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
kpmg_df['clean_pros'] = kpmg_df['pros'].apply(clean_text)

#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(kpmg_df["clean_pros"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
kpmg_df["Topic"] = topic_results.argmax(axis=1)

kpmg_df.head()


###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)



#################################################################################################################################
### PART 3.2: Dominant Topics in Reviews - Cons
#################################################################################################################################

#Import dataset 
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import spacy

df = pd.read_csv('/Users/samiabelmadani/Downloads/df_big4.csv')

####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|lot|company|consulting|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
df['clean_cons'] = df['cons'].apply(clean_text)
#Remove empty rows
df = df[df['clean_cons'].str.strip() != '']

#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(df["clean_cons"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
df["Topic"] = topic_results.argmax(axis=1)

df.head()

###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)

##=========DELOITE###########
# Filter the DataFrame to include only rows where the "firm" column equals "EY"
deloitte_df = df[df['firm'] == 'Deloitte']

####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|lot|company|consulting|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
deloitte_df['clean_cons'] = deloitte_df['cons'].apply(clean_text)
#Remove empty rows
deloitte_df = deloitte_df[deloitte_df['clean_cons'].str.strip() != '']


#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(deloitte_df["clean_cons"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
deloitte_df["Topic"] = topic_results.argmax(axis=1)

deloitte_df.head()

###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)

##=========PwC###########
# Filter the DataFrame to include only rows where the "firm" column equals "EY"
pwc_df = df[df['firm'] == 'PwC']

####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|lot|company|consulting|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
pwc_df['clean_cons'] = pwc_df['cons'].apply(clean_text)
#Remove empty rows
pwc_df = pwc_df[pwc_df['clean_cons'].str.strip() != '']


#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(pwc_df["clean_cons"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
pwc_df["Topic"] = topic_results.argmax(axis=1)

pwc_df.head()

###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)

##=========EY###########
# Filter the DataFrame to include only rows where the "firm" column equals "EY"
ey_df = df[df['firm'] == 'EY']

####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|lot|company|consulting|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
ey_df['clean_cons'] = ey_df['cons'].apply(clean_text)
#Remove empty rows
ey_df = ey_df[ey_df['clean_cons'].str.strip() != '']


#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(ey_df["clean_cons"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
ey_df["Topic"] = topic_results.argmax(axis=1)

ey_df.head()

###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)

##=========KPMG###########
# Filter the DataFrame to include only rows where the "firm" column equals "EY"
kpmg_df = df[df['firm'] == 'KPMG']

####################
###Pre-processing###
####################

nlp = spacy.load('en_core_web_sm')

import re

def clean_text(x):
    # Convert to lowercase
    x = x.lower()
    # Remove new line
    x = re.sub(r'\n', ' ', x)
    # Remove links
    x = re.sub(r"https?://[^\s]+", "", x)
    # Remove hashtags
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x)
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc|lot|company|consulting|like', ' ', x)
    # Remove bullet points preceded by words
    x = re.sub(r'(\w)-', r'\1 ', x)
    # Remove numbers followed by periods
    x = re.sub(r'\b\d+\.\s+', '', x)
    # Remove commas
    x = re.sub(r',', '', x)
    # Remove stop words
    stop_words = ["》", "•", "/", "|", ".", "?", "big 4", "big4", "big-4", "big 4's", ";", ":", "!", 
                  "...", "e.g.", "#", "*", "+", "-", "\"", "1.", "1-", ">", "1)", "%", "$", "&"]
    for word in stop_words:
        x = x.replace(word, '')
    # Remove numbers attached to words
    x = re.sub(r'\b\d+\b', '', x)
    # Remove word "work" and its related forms
    x = re.sub(r'\bwork\w*\b', '', x)
    # Lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # Remove multiple spaces
    x = re.sub(r' +', ' ', x)
    # Remove leading and trailing spaces
    x = x.strip()
 
    return x

# Example usage
kpmg_df['clean_cons'] = kpmg_df['cons'].apply(clean_text)
#Remove empty rows
kpmg_df = kpmg_df[kpmg_df['clean_cons'].str.strip() != '']


#########
###LDA###
#########

# We indicate that we would like to exclude any English stopwords, words that show up in less than 2 documents
# and words that are common across 90% of the documents in the corpus since these words would not help with distinguishing the documents.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")

dtm = cv.fit_transform(kpmg_df["clean_cons"])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

# We train the model.
lda.fit(dtm)

len(cv.get_feature_names())

len(lda.components_[0])

# Print the top 15 words in each topic
n = 4

for index, topic in enumerate(lda.components_):
    print(f'The top {n} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-n:]])

topic_results = lda.transform(dtm)

print(topic_results[0].round(2))

# Get the highest probability in the topic distribution and assign it as the topic of the document
kpmg_df["Topic"] = topic_results.argmax(axis=1)

kpmg_df.head()

###############
###Bar Plot####
###############

import matplotlib.pyplot as plt

# Define a function to visualize topics using bar plots
def visualize_topics(lda_model, cv_model, num_words=4):
    # Get the feature names (words) from the CountVectorizer
    feature_names = cv_model.get_feature_names()

    # Plot bar plots for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        # Sort words by their weights in the topic
        sorted_word_indices = topic.argsort()[:-num_words-1:-1]
        sorted_words = [feature_names[i] for i in sorted_word_indices]
        sorted_weights = [topic[i] for i in sorted_word_indices]

        # Plot bar plot
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_words, sorted_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.ylabel('Word')
        plt.title(f'Topic #{topic_idx}', fontsize=20)
        plt.gca().invert_yaxis()  # Invert y-axis to display top words at the top
        plt.show()

# Call the function to visualize topics
visualize_topics(lda, cv)

###########################
###Hyperparameter Tuning###
###########################
from sklearn.model_selection import GridSearchCV

# Define the range of hyperparameters to search
param_grid = {'n_components': [5, 7, 10, 15]}  # Adjust the values as needed

# Initialize LDA model
lda = LatentDirichletAllocation(random_state=42)

# Perform grid search
grid_search = GridSearchCV(lda, param_grid, cv=5, n_jobs=-1)
grid_search.fit(dtm)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best LDA model
best_lda_model = grid_search.best_estimator_

# Print the top words for each topic in the best LDA model
num_words = 4  # Adjust as needed
for index, topic in enumerate(best_lda_model.components_):
    print(f'The top {num_words} words for topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-num_words:]])

# Visualize the topics using bar plots
visualize_topics(best_lda_model, cv)



#################################################################################################################################
### PART 4: Personalized Recommendation
#################################################################################################################################

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import warnings
import re

import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
from spacy.lang.en import stop_words

#to ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df_big4_sim = pd.read_csv("C:\\Users\\olivi\\OneDrive\\Documents\\McGill MMA\\Winter 2024\\Text Analytics (INSY 669)\\Final project\\df_big4.csv")

# keep useful columns only
columns_to_keep = ['firm', 'pros']
df_big4_sim = df_big4_sim[columns_to_keep].copy()
df_big4_sim = df_big4_sim.reset_index(drop=True)

#checking for nulls if present any
print("Number of rows with null values:")
print(df_big4_sim.isnull().sum().sum())
df_big4_sim=df_big4_sim.dropna()

# Equilibrate the dataset with 1000 reviews of each type
# Assuming df_big4_sim is your existing DataFrame
selected_firms = ['Deloitte', 'KPMG', 'EY', 'PwC']
num_rows_to_keep = 1000
min_characters = 40

# Create a mask to filter rows for the specified firms and with at least 50 characters in 'pros'
mask = (df_big4_sim['firm'].isin(selected_firms)) & (df_big4_sim['pros'].str.len() >= min_characters)

# Use groupby and sample to randomly select 1000 rows for each firm
df_big4_sim = df_big4_sim[mask].groupby('firm', group_keys=False).apply(lambda x: x.sample(min(len(x), num_rows_to_keep)))

# Reset the index to have a continuous index for the resulting DataFrame
df_big4_sim = df_big4_sim.reset_index(drop=True)


# user input
#######################################################################################
looking_for = str(input("Indicate the qualities you are searching for in a company: "))
# for test puroposes try the following:
# "Good salary, work-life balance, opportunities, benefits, training"
#######################################################################################

#merging looking_for to the reviews
tempDataFrame=pd.DataFrame({'criterias':[looking_for]})
tempDataFrame=tempDataFrame.transpose()
description_list1=df_big4_sim['pros']
frames = [tempDataFrame, description_list1]
result = pd.concat(frames)
result.columns = ['review']
result=result.reset_index()

#clean text
nlp = spacy.load('en_core_web_sm')
def clean_text(x): # x is a document
    # conver to lower case 
    x = x.lower()
    # remove new line
    x = re.sub(r'\n', ' ', x)
    # remove link
    x = re.sub(r"https?://[^\s]+", "", x)
    # remove name twitter tag
    x = re.sub(r'@[A-Za-z0-9\_]+', ' ', x) 
    # remove name hashtag tag
    x = re.sub(r'\#[A-Za-z0-9\_]+', ' ', x) 
    # Remove specific words - Deloitte, EY, KPMG, PwC
    x = re.sub(r'deloitte|ey|kpmg|pwc', ' ', x)
    # lemmatization
    x = ' '.join([token.lemma_ for token in nlp(x)])
    # remove multiple space
    x = re.sub(r' +', ' ', x)
    # remove space before and after the text
    x = x.strip()
    return x

#building bag of words using frequency
vec_words = CountVectorizer(decode_error='ignore', 
                            stop_words=list(stop_words.STOP_WORDS), 
                            preprocessor=clean_text)

total_features_words = vec_words.fit_transform(result['review'])

#Calculating pairwise cosine similarity
subset_sparse = sparse.csr_matrix(total_features_words)
total_features_review=subset_sparse
total_features_attr=subset_sparse[0,]
similarity=1-pairwise_distances(total_features_attr,total_features_review, metric='cosine')

#Assigning the similarity score to dataframe
#similarity=np.array(similarities[0]).reshape(-1,).tolist()
similarity=pd.DataFrame(similarity)
similarity=similarity.transpose()
similarity.columns = ['similarity']
similarity=similarity.drop(similarity.index[[0]])
df_big4_sim=df_big4_sim.assign(similarity=similarity.values)

##############################

# Reorder in descending order of similarity score
df_big4_sim = df_big4_sim.sort_values(by='similarity', ascending=False).reset_index(drop=True)

# select only the top 100 scores
df_big4_sim_top100 = df_big4_sim.head(100)

# frequency of firms in the top 100
firm_frequency = df_big4_sim_top100['firm'].value_counts()

# Find the firm with the highest frequency in the top 100 rows
best_fit = df_big4_sim_top100['firm'].value_counts().idxmax()

# show best fit
print(f"Based on your preferences, the company that is the best fit is {best_fit}")





