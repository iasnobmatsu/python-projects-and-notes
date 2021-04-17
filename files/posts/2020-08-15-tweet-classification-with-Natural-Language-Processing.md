---
layout: post
title:  "Tweet Classification with NLP"
date:   2020-08-15
excerpt: "text cleaning using regular expressions, NLP visuLization with wordclouds, naive bayes, logistic regression, and LSTM models for binary text classification in python"
---

<a id='top'></a>
# Introduction

<hr>

Twitter is a popular social network service. Because of the accessibility and universality of Twitter, people are starting to tweet about disasters in emergencies. In this notebook, we will attempt to categorize tweets as either disaster tweets (target=1) or non-disasrer tweets (target=0) for the *Real or Not? NLP with Disaster Tweets* challenge using natural language processing in python.

This is my kaggle kernal from [here](https://www.kaggle.com/iasnobmatsu/nlp-starter-guide-with-multiple-models).




#### table of contents
* [import libraries & datasets](#import)
    * [import libraries](#library)
    * [look at datasets](#dataset)
    * [Evaluation Criteria](#)
* [exploratory data analysis](#EDA)
    * [target](#target)
    * [location](#location)
    * [keyword](#keyword)
    * [text](#text)
        * [number of characters](#char)
        * [number of words](#word)
        * [word length](#wlen)
        * [word clouds](#cloud)
* [data preprocessing](#preprocess)
* [Evaluation criteria and notes](#eval)
    * [f1 score](#f1)
    * [cross validation](#cross)
* [modeling](#model)
    * [naive bayes](#nb)
    * [logistic regression](#lr)
    * [LSTM](#LSTM)

#### Acknowledgements

This notebook is inspired by and contains ideas from the following kaggle kernels. If you find this notebook helpful, please check out the following as well.

* [NLP - EDA, Bag of Words, TF IDF, GloVe, BERT](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert) 
* [📙 CheatSheet: Text Data](https://www.kaggle.com/prestonfan/cheatsheet-text-data)
* [Getting started with NLP - A general Intro](https://www.kaggle.com/parulpandey/getting-started-with-nlp-a-general-intro) 
* [Basic EDA,Cleaning and GloVe](https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove) 
* [Start From Here : Disaster Tweets EDA+Basic model](https://www.kaggle.com/ratan123/start-from-here-disaster-tweets-eda-basic-model)
* [Disaster Tweets: EDA | NLP | Classifier Models](https://www.kaggle.com/kushbhatnagar/disaster-tweets-eda-nlp-classifier-models)
* [Keras LSTM](https://www.kaggle.com/adamlouly/simple-keras-lstm-for-warming-up)
* [A Detailed Explanation of Keras Embedding Layer](https://www.kaggle.com/rajmehra03/a-detailed-explanation-of-keras-embedding-layer)
* [LSTM baseline](https://www.kaggle.com/bibek777/lstm-baseline)


[back to top](#top)

<a id="import"></a>
#  Import Libraries and Datasets

<hr>

[back to top](#top)

<a id='library'></a>
### Import Libraries

First, we will import all libraries that will be used. Here are some short introductions of all the used libraries.

| Library     | Description and link to documentation |
| :---------- | :---------- |
| re          | [regular expression operations](https://docs.python.org/3/library/re.html)                     |
| numpy       | [linear algebra and data manipulation](https://numpy.org/doc/stable/)                          |
| pandas      | [data processing, csv manipulation ](https://pandas.pydata.org/docs/)                          |
| matplotlib  | [basic data visualization](https://matplotlib.org/users/index.html)                            |
| scipy       | [more various data visualization](https://seaborn.pydata.org/tutorial.html)                    |
| string      | [string and text operations](https://docs.python.org/3/library/string.html)                    |
| nltk        | [natural language processing](https://www.nltk.org/)                                           |
| wordcloud   | [visualizing word clouds](https://amueller.github.io/word_cloud/auto_examples/index.html)      |
| sklearn     | [machine learning](https://scikit-learn.org/stable/)                                           |
| keras       | [neural networks and deep learning](https://keras.io/guides/)                                  |



```python
import re 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,SpatialDropout1D,Embedding
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
```

[back to top](#top)

<a id='dataset'></a>
### First Look at Datasets
Now we will import train and test datasets and take an initial look at the shapes of the datasets. 

The training set has 7613 rows and 5 columns (id, keyword, location, text, and target).
- id: a unique identifier for each tweet
- text: the text of the tweet
- location: the location the tweet was sent from (may be blank)
- keyword: a particular keyword from the tweet (may be blank)
- target: in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

The test set has 3243 rows and 4 columns (id, keyword, location, and text).
- id: a unique identifier for each tweet
- text: the text of the tweet
- location: the location the tweet was sent from (may be blank)
- keyword: a particular keyword from the tweet (may be blank)

The test dataset does not contain the target variable. This need to be predicted.




```python
#import disaster tweets dataframes
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')
df_test = pd.read_csv('../input/nlp-getting-started/test.csv')

#take a look at dataset sizes
print('train shape:', df_train.shape)
print('test shape:',df_test.shape)
```

    train shape: (7613, 5)
    test shape: (3263, 4)



```python
#take a look at the train dataset
pd.set_option('display.max_colwidth', 300) #set width of columns to display full tweet
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation orders in California</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#take a look at the test dataset
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, stay safe everyone.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are fleeing across the street, I cannot save them all</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>



[back to top](#top)

<a id="EDA"></a>
#  Exploratory Data Analysis 

<hr>

<a id='target'></a>
### Target

How many tweets are disaster tweets and how many are non disaster tweets? The following cell will compute these numbers and display proportions using a pie chart.

|type| count | type | count|
|------------|------|---------|-----|
|Non disaster| 4342 | Disaster| 3271|


```python
target_count=df_train['target'].value_counts(dropna=False) #count target
plt.figure(figsize=(5,5)) #set figure size
plt.pie([target_count[0], target_count[1]],labels=['not disaster', 'disaster'], shadow=False)#pie chart
```




    ([<matplotlib.patches.Wedge at 0x7efa90569050>,
      <matplotlib.patches.Wedge at 0x7efa90569590>],
     [Text(-0.24110481617711207, 1.0732513534192263, 'not disaster'),
      Text(0.24110481617711216, -1.0732513534192263, 'disaster')])




![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_11_1.png)



[back to top](#top)

<a id='location'></a>
### Location

Now we can explore the distribution of the location variable (location when tweet is posted) stratified by target.
* Locations are not mutually exclusive from each other i.e. USA would include NY etc.
* The distribution of most freqyebt locations differed by target (1/0).
* Although the visualization do not show missing values, there is a large proportion of missing values for location.


```python
# percent of location appearing grouped by target = 1/0
location_count_1=pd.DataFrame(df_train[df_train['target']==1]['location'].value_counts(dropna=False)) #find only disaster tweets
location_count_1 = location_count_1.reset_index() #reformate
location_count_1.columns=['location','count'] #rename headers
location_count_1['percent']=location_count_1['count']/location_count_1['count'].sum() #percentage

location_count_0=pd.DataFrame(df_train[df_train['target']==0]['location'].value_counts(dropna=False)) #only non-disaster
location_count_0 = location_count_0.reset_index() #reformat
location_count_0.columns=['location','count'] #headers
location_count_0['percent']=location_count_0['count']/location_count_0['count'].sum() #percentage

#make separate bar charts for taget =1/0
fig,a =  plt.subplots(2,1,figsize=(15,10)) #make 2 subplots 
fig.tight_layout(pad=12) #padding between subplots
print('number of different locations (disaster):', location_count_1.shape[0]) 
sns.barplot(x='location',y='count', data=location_count_1[:20], palette='Spectral', ax=a[0]) # barplot for top 20 most common 
a[0].set_title('target=1') 
a[0].tick_params(labelrotation=45)

print('number of different locations (non disaster):', location_count_0.shape[0])
sns.barplot(x='location',y='count', data=location_count_0[:20], palette='Spectral', ax=a[1]) # barplot for top 20 most common 
a[1].set_title('target=0')
a[1].tick_params(labelrotation=45)

print(location_count_0.head(1))
print(location_count_1.head(1))
```

    number of different locations (disaster): 1514
    number of different locations (non disaster): 2143
      location  count  percent
    0      NaN   1458  0.33579
      location  count   percent
    0      NaN   1075  0.328646



![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_13_1.png)


[back to top](#top)

<a id='keyword'></a>
### Keyword

Similarly, we can look at keyword distribution.
* The distrbution of keywords also differed by target.
* There are not as much missing values as location but missing values still exist. 


```python
# percent of keywords appearing grouped by target = 1/0
key_count_1=pd.DataFrame(df_train[df_train['target']==1]['keyword'].value_counts(dropna=False)) # only disaster
key_count_1 = key_count_1.reset_index()
key_count_1.columns=['keyword','count']
key_count_1['percent']=key_count_1['count']/key_count_1['count'].sum()

key_count_0=pd.DataFrame(df_train[df_train['target']==0]['keyword'].value_counts(dropna=False)) #only non disaster
key_count_0 = key_count_0.reset_index()
key_count_0.columns=['keyword','count']
key_count_0['percent']=key_count_0['count']/key_count_0['count'].sum()


#make separate bar charts for taget =1/0
fig,a =  plt.subplots(2,1,figsize=(15,10)) #make 2 subplots for target=1, target=0
fig.tight_layout(pad=12)
print('number of different keywords (disaster):', key_count_1.shape[0])
sns.barplot(x='keyword',y='count', data=key_count_1[:20], palette='Spectral', ax=a[0]) # barplot for top 20 most common 
a[0].set_title('target=1')
a[0].tick_params(labelrotation=45)

print('number of different keywords (non disaster):', key_count_0.shape[0])
sns.barplot(x='keyword',y='count', data=key_count_0[:20], palette='Spectral', ax=a[1]) # barplot for top 20 most common 
a[1].set_title('target=0')
a[1].tick_params(labelrotation=45)
```

    number of different keywords (disaster): 221
    number of different keywords (non disaster): 219



![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_15_1.png)


[back to top](#top)

<a id='text'></a>
### Text

For natural language processing, text EDA tast is vital. Some commonly used text features to explore are word frequencies, word length, sentence length etc. Here, we will explore 3 features--number of characters, number of words, and word length. We will also complete a wordcloud to visualize the texts.

More information on EDA for NLP can be found on this blog https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools. 

<a id='char'></a>
#### ***Number of characters***

Now let's explore features of the tweet texts stratified by target. We will start with number of characters. 

It is quite hard to tell whether the distributions differ, so I used a t-test to see if the distributions of tweet characters differ based on target. T-test assumes means of samples to be normal, for the tweet sample size, I believe it is okay to run a t-test based on CLT. 
* The result of the t-test gives t=16.13 and p<0.05. Number of characters does differ by target.   


```python
# make a new variable for number of characters
df_train['characters']=df_train['text'].str.len()

# split dataset by target
char_1=df_train[df_train['target']==0]['characters']
char_0=df_train[df_train['target']==1]['characters']

# t test
clengtht, clengthp=scipy.stats.ttest_ind(char_1, char_0)
print('T Test number of characters by target t={:.2f},p={:.2f}'.format(clengtht, clengthp))

#histograms
fig,a =  plt.subplots(1,3,figsize=(15,5)) #make 3 subplots for target=1, target=0, and complete sample
sns.distplot(char_1,ax=a[0], color='purple')
a[0].set_title('target = 1')
sns.distplot(char_0,ax=a[1], color='blue')
a[1].set_title('target = 0')
sns.distplot(df_train['characters'],ax=a[2], color='green')
a[2].set_title('target = 0 and target = 1')
```

    T Test number of characters by target t=-16.13,p=0.00





    Text(0.5, 1.0, 'target = 0 and target = 1')




![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_17_2.png)


[back to top](#top)

<a id='word'></a>
#### ***Number of words***

Similarly, we can explore number of words in tweets.

* The result of the t-test gives t=-3.49 and p<0.05. Number of characters does differ by target.   


```python
# make a new variable for number of words
df_train['words']=df_train['text'].apply(lambda x: len(str(x).split())) #split by space to turn tweet into words

# split dataset by target
w_1=df_train[df_train['target']==0]['words']
w_0=df_train[df_train['target']==1]['words']

# t test
wlengtht, wlengthp=scipy.stats.ttest_ind(w_1, w_0)
print('T Test number of words by target t={:.2f},p={:.2f}'.format(wlengtht, wlengthp))

#histograms
fig,a =  plt.subplots(1,3,figsize=(15,5)) #make 3 subplots for target=1, target=0, and complete sample
sns.distplot(w_1,ax=a[0], color='purple')
a[0].set_title('target = 1')
sns.distplot(w_0,ax=a[1], color='blue')
a[1].set_title('target = 0')
sns.distplot(df_train['words'],ax=a[2], color='green')
a[2].set_title('target = 0 and target = 1')
```

    T Test number of words by target t=-3.49,p=0.00





    Text(0.5, 1.0, 'target = 0 and target = 1')




![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_19_2.png)



[back to top](#top)

<a id='wlen'></a>
#### ***word length in tweets***

The last feature we are exploring here is length of tweets. 

* t=-15.68,p=0.00,so the word length distribution differs by target.


```python
# make a new variable for number of characters
df_train['wlen']=df_train['text'].apply(lambda x: sum([len(a) for a in str(x).split()])/len(str(x).split()))
#split by space to turn tweet into words, use list comprehension to get total char length, divide by word list length

# split dataset by target
wl_1=df_train[df_train['target']==0]['wlen']
wl_0=df_train[df_train['target']==1]['wlen']

# t test
wllengtht, wllengthp=scipy.stats.ttest_ind(wl_1, wl_0)
print('T Test number of words by target t={:.2f},p={:.2f}'.format(wllengtht, wllengthp))

#histograms
fig,a =  plt.subplots(1,3,figsize=(15,5)) #make 3 subplots for target=1, target=0, and complete sample
sns.distplot(wl_1,ax=a[0], color='purple')
a[0].set_title('target = 1')
sns.distplot(wl_0,ax=a[1], color='blue')
a[1].set_title('target = 0')
sns.distplot(df_train['wlen'],ax=a[2], color='green')
a[2].set_title('target = 0 and target = 1')
```

    T Test number of words by target t=-15.68,p=0.00





    Text(0.5, 1.0, 'target = 0 and target = 1')




![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_21_2.png)


[back to top](#top)

<a id='cloud'></a>
#### ***word clouds***
Word clouds are visualizations of words in which the sizes of words reflect the relative importance of words. Here, we will build word clouds using the raw text data. We can also choose to build word clouds after cleaning the text data and getting rid of noises (punctuations, lines, etc)

This tutorial here walks through steps of building word clouds. https://www.datacamp.com/community/tutorials/wordcloud-python.

The generated word clouds, although containing some noise such as 'https' does show that words makeup of disaster tweets look differetly from words makeup of non disaster tweets such that the disaster tweet word cloud includes words such as 'suicide','bomber','building', whereas the non disaster tweet word cloud includes words such as 'good', 'great', and 'love'.

From the word clouds, I think a good extra step to take is to do a simple sentiment analysis and explore the emotions of the disaster vs non disaster tweets. I have not completed this step in this notebook yet and might add it later. 


```python
# putting all texts across rows together as a big string variable 
alltextdisaster=' '.join(set([text for text in df_train[df_train['target']==1]['text']])) # disaster
alltextnondisaster=' '.join(set([text for text in df_train[df_train['target']==0]['text']])) # non disaster

# build word clouds 
wc1 = WordCloud(background_color="white", max_words=200, width=1000, height=800).generate(alltextdisaster)
wc2 = WordCloud(background_color="white", max_words=200, width=1000, height=800).generate(alltextnondisaster)

# plotting word clouds
fig,a =  plt.subplots(1,2,figsize=(20,10))
a[0].imshow(wc1, interpolation='bilinear')
a[0].axis("off")
a[0].set_title('disaster tweet word cloud')

a[1].imshow(wc2, interpolation='bilinear')
a[1].axis("off")
a[1].set_title('nondisaster tweet word cloud')
plt.show()
```


![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_23_0.png)


[back to top](#top)

<a id="preprocess"></a>
#  Data Preprocessing 

<hr>

Text processing is an important step in NLP. Here we will compile a cleanTextData() function that removes punctuations, numbers, html tags, urls, and emojis. This step also turns everything into lowercase. Removing stopwrods and stemming are also common techniques used in NLP data cleaning. 

Stopwords are words often ignored in text processing and the ommission of which does not generally change meanings of texts. Here, we are using stopwords in the nltk library, which included 'i', 'me', 'as', 'until', 'so', 'than', 'too', 'very' etc.

Stemming is the process of converting words into their root forms. This step is helpful because there could be variants of the same word (eat and eating) in the text. We are using the nltk porterstemmer here to stem the words which follows algorithms by Porter (1980).

Note that list comprehension and regular expressions are also used here in data preprocessing. There are other libraries which can automatically process punctuations, numbers, html tags etc as well.


1. List comprehension: list comprehension is a simpler and more concise way create lists. The format of list comprehension is `[ statement to generate list ]`.

    Example: `templist=[c for c in text if c not in string.punctuation]`

    This is the same as: 
    ```
    templist=[]
        for c in text:
            if c not in string.punctuation:
                templist.append(c)
    ```
    
2. Regular expression: regular expressions are text sequence pattern matching blocks. 
    * re.sub(pattern, replacement, string): replace *pattern* in *string* with *replacement*
    * re.complile(pattern): compile the *pattern* into a regular expression object which could later be used to match
    * r'pattern': convert *pattern* to raw string, which does not compile escape sequences i.e. '\n'
    * some regular expression operators used in the notebook:
        * `?`  match 0 or 1 instance of the charater preceding. i.e. 123? will match 12 or 123
        * `.`  match any character
        * `*` match as many consecutive instances of preceding character as possible
        * `.*?`: match as few characters as possible
        * `+`: match 1 or more of the preceding character
        * `\S`: match any non white space character
        * `1|2` : match 1 or 2
        * `\` : escape characters
    * `r'<.\*?>'` will match `<anytext>` minimally, so for `'<a></a>'`, the pattern matches `'<a>'` and `'<a/>'` separately instead of matching the whole thing   
    * `r'https?://\S+|www\.\S+'` will match `http(s)://www.AnyNonWhiteSpaceText` and `http(s)://AnyNonWhiteSpaceText`
    * you can test regular expressions online using websites such as https://regex101.com/.




```python
def removePunctuation(text):
    return "".join([c for c in text if c not in string.punctuation])
print('remove punctuation:', removePunctuation("It's me!!!! :/"))

def removeNumber(text):
    return "".join([c for c in text if not c.isdigit()])
print('remove numbers:', removeNumber("123 abc"))

def removeHTML(text):
    return re.sub(r'<.*?>','', text) # match <tag> minimally
print('remove HTML tags:', removeHTML("<h1>heading</h1><p attribute=''>tag"))

def removeURL(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text) # match url patterns
print('remove url:', removeURL("url https://www.kaggle.com kaggle"))

def removeEmoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE) # compiling all emojis as a reg ex expression
    return emoji_pattern.sub(r'', text)
print('remove emoji:', removeEmoji('Sad😔'))

def lowerCase(text): 
    return text.lower()
print('lower case:', lowerCase('crazy NoiSy Town!'))

def removeStopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])
print('remove stop words:', removeStopwords('I am a cup of tea'))


Pstemmer=PorterStemmer()
def stemText(text):
    return ' '.join([Pstemmer.stem(token) for token in text.split()])
print('stem Text:', stemText('Word clouds are visualizations of words in which the sizes of words reflect the relative importance of words'))


# put all the above cleaning functions into one function
def cleanTextData(text):
    text=lowerCase(text)
    text=removePunctuation(text)
    text=removeURL(text)
    text=removeEmoji(text)
    text=removeNumber(text)
    text=removeHTML(text)
    text=removeStopwords(text)
    text=stemText(text)
    return text
print('clean:', cleanTextData('Word clouds are visualizations of words in which the sizes of words reflect the relative importance of words <a>link https://www.kaggle.com<a/> ttps://www.kaggle.com 321😔'))

#clean train and test
df_train['cleaned_text']=df_train['text'].apply(lambda x: cleanTextData(x))
df_test['cleaned_text']=df_test['text'].apply(lambda x: cleanTextData(x))
df_train.head(10)
```

    remove punctuation: Its me 
    remove numbers:  abc
    remove HTML tags: headingtag
    remove url: url  kaggle
    remove emoji: Sad
    lower case: crazy noisy town!
    remove stop words: I cup tea
    stem Text: word cloud are visual of word in which the size of word reflect the rel import of word
    clean: word cloud visual word size word reflect rel import word alink httpswwwkagglecoma ttpswwwkagglecom





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
      <th>characters</th>
      <th>words</th>
      <th>wlen</th>
      <th>cleaned_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</td>
      <td>1</td>
      <td>69</td>
      <td>13</td>
      <td>4.384615</td>
      <td>deed reason earthquak may allah forgiv us</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
      <td>38</td>
      <td>7</td>
      <td>4.571429</td>
      <td>forest fire near la rong sask canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</td>
      <td>1</td>
      <td>133</td>
      <td>22</td>
      <td>5.090909</td>
      <td>resid ask shelter place notifi offic evacu shelter place order expect</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation orders in California</td>
      <td>1</td>
      <td>65</td>
      <td>8</td>
      <td>7.125000</td>
      <td>peopl receiv wildfir evacu order california</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school</td>
      <td>1</td>
      <td>88</td>
      <td>16</td>
      <td>4.500000</td>
      <td>got sent photo rubi alaska smoke wildfir pour school</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>#RockyFire Update =&gt; California Hwy. 20 closed in both directions due to Lake County fire - #CAfire #wildfires</td>
      <td>1</td>
      <td>110</td>
      <td>18</td>
      <td>5.166667</td>
      <td>rockyfir updat california hwi close direct due lake counti fire cafir wildfir</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>#flood #disaster Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas</td>
      <td>1</td>
      <td>95</td>
      <td>14</td>
      <td>5.857143</td>
      <td>flood disast heavi rain caus flash flood street manit colorado spring area</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I'm on top of the hill and I can see a fire in the woods...</td>
      <td>1</td>
      <td>59</td>
      <td>15</td>
      <td>3.000000</td>
      <td>im top hill see fire wood</td>
    </tr>
    <tr>
      <th>8</th>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>There's an emergency evacuation happening now in the building across the street</td>
      <td>1</td>
      <td>79</td>
      <td>12</td>
      <td>5.666667</td>
      <td>there emerg evacu happen build across street</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I'm afraid that the tornado is coming to our area...</td>
      <td>1</td>
      <td>52</td>
      <td>10</td>
      <td>4.300000</td>
      <td>im afraid tornado come area</td>
    </tr>
  </tbody>
</table>
</div>



[back to top](#top)


<a id='eval'></a>
## Evaluation and Notes

<a id='f1'></a>
### F1 score
Sklearn's f1 scoring criteria will be used to evaluate models. F1 score is the evaluation criterion specified by the *Real or Not? NLP with Disaster Tweets* challange. As pointed out in the challenge, f1 can be calculated as $F1=2∗\frac{precision∗recall}{precision+recall}$. 

* Precision is defined as $\frac{TP}{TP+FP}$.
* Recall is defined as $\frac{TP}{TP+FN}$.
* TF(True Positive) means both prediction and actual label are 1.
* TN(True Negative) means both prediction and actual label are 0.
* FP(False Positive) means prediction is 1 but true label is 0.
* FN(False Negative) means prediction is 0 but true label is 1.

Often in research, TF, TN, FP, FN are related to type I and type II errors. FP is a false alert and is refered to as type I error and is related to statistical significance alpha, FN is referred to as type II error and is related to statistical power beta.

(off topic) An interesting article about statistical significance and power above to read is Cohen (1990) - The earth is round (p<0.05).



### Cross Validation

Cross Validation validates an algorithm's performance through sampling. K-fold cross validation will split the randomized sample into k groups. When validating, there will be k iterations, for each iteration, a group will be used for testing whereas the remaining  k-1 groups will be used for training. Sklearn's 10 fold cross validation is used in the following models to get f1 accuracies of the training dataset. 


[back to top](#top)

<a id="model"></a>
#  Modeling 

Now we can finally start to build models! (yay!)

We will compare the efficacies of different algorithms in this section. The algorithms compared are:
* Naive bayes
* Logistic regression
* LSTM recurrent neural network

[back to top](#top)

<a id='tf'></a>

### TF-IDF

We need to vectorize the texts before using them in the models. sklearn's CountVectorizer and TfidfVectorizer (Tfidf stands for term frequency inverse document frequency) are both good choices to vectorize words based on term frequencies. The difference, put in simple words, is that TfidfVectorizor normalizes the count matrix after counting term frequencies. We will use TFidfVectorizer here. 

Term frequency $TF$ is the number of times the term occured in a document. 

Inverse document frequency can be calculated as $IDF = log(\frac{Total Number Of Documents}{Frequency Of Term/NumberOfDocumentsWithTerm})+ 1 $. 

Finally,TF-IDF can be calculated as $TF * IDF$.

We can run a simple example to see how countvectorizer and tfidfvectorizer works.

The following code uses the couuntctorizer to derive frequencies for 'cat runs' and 'dog runs'. 

```
vectorizer=CountVectorizer()
vectors = vectorizer.fit_transform(['cat runs', 'dog runs'])
print(vectorizer.get_feature_names(),vectors.toarray())
```
printing the results shows

|           | cat    | dog    | runs   |  
|-----------|--------|--------|--------|
|cat runs   | 1      | 0      | 1      | 
|dog runs   | 0      | 1      | 1      | 

<br>
<br>
Similarly, the following code uses the tfidfvectorizer to derive frequencies for 'cat runs' and 'dog runs'. 

```
vectorizer=TfidfVectorizer()
vectors = vectorizer.fit_transform(['cat runs', 'dog runs'])
print(vectorizer.get_feature_names(),vectors.toarray())
```
printing the results shows 

|           | cat    | dog    | runs   |  
|-----------|--------|--------|--------|
|cat runs   | 0.8148 | 0      | 0.5797 | 
|dog runs   | 0      | 0.8148 | 0.5797 | 

<br>
<br>
It is quite obvious that CountVectorizer and TfidfVectorizers have similar mechanisms although countvectorizer returns integers whereas tfidfvectorizer returns floats. TfidfVectorizer has the benefit of avoiding putting too much weights on frequently appearing words by making the encoding in inverse proportion to the frequencies. 



```python
train_vectors=TfidfVectorizer().fit_transform(df_train['cleaned_text'])
test_vectors=TfidfVectorizer().fit_transform(df_test['cleaned_text'])
y=df_train['target']
X=train_vectors
```

[back to top](#top)


<a id='nb'></a>
### Naive Bayes Model

#### ***Theory behind naive bayes***

The first model we are trying out is the naive bayes model. The naive bayes model is based on the bayes theorem which links the probability of event C given event X to the probabily of event X given event C. The theorem states $P(C|X)=\frac{P(X|C)P(C)}{P(X)}$ which is derived from conditional probability $P(X|C)=\frac{P(X \cap C)}{P(C)}$ => $P(C \cap X)=P(X|C)*P(C)$ => $P(C|X)=\frac{P(C \cap X)}{P(X)}=\frac{P(X|C)P(C)}{P(X)}$.
* P(C) is the probability of C
* P(X) is the probability of X
* P(C&#124;X) is the conditional probability of C given X
* P(X&#124;C) is the conditional probability of X given C

In building a classifier, we can view X as the feature and C as the target we are trying to classify. We want to find out probability of the variable to be classified given the occurence of the features. Because usually there are many features i.e. $X_1,X_2,X_3$ etc, the equation can be expanded to be $P(C&#124;X_1,X_2,X_3...)=\frac{P(X_1 \cap X_2 \cap X_3...&#124;C)P(C)}{P(X_1 \cap X_2 \cap X_3...)}$.

The above equation will be very hard to calculate, but naive bayes assumers features $X_1,X_2,X_3$ etc to be independent, meaning that the occurance of the features do not affect each other. With this assumption of indepdendence, the equation becomes $P(C&#124;X_1,X_2,X_3...)=\frac{P(X_1&#124;C)P(X_2&#124;C)P(X_3&#124;C)...P(C)}{P(X_1)P(X_2)P(X_3)...}$.

We often have two classes to classify the target into, i.e. in this example, disaster tweet vs non disaster tweet. Using the above naive bayes equation, $P(disaster&#124;features)$ and $P(non-disaster&#124;features)$ can be calculated because $P(X_1&#124;C)$ and $P(X_1)$ etc can be derived with training data. The class label that should be assigned to the data of interest should the class with a larger probability.

References:
* [Naive Bayes Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA) 
* [naive bayes classifiers](https://www.geeksforgeeks.org/naive-bayes-classifiers/)
* [Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)

#### ***Using naive bayes***

In application, multinomial and guassian are two popular forms of naive bayes. The difference of the two comes from the assumption of feature distributions. As the names implied, multinomial naive bayes assumes a multinomial (discrete) probability mass function of data, whereas gaussian naive bayes assumes a gaussian (continuous) probability distribution of data. 

We will use sklearns to run both a multinomial naive bayes model and a gaussian naive bayes model which is fairly simple to run. We will also use sklearn's cross validation, and f1 scoring metric. A confusion matrix also helps visualizing algorithm performance.

#### ***Multinomial NB***


```python
#Multinomial NB
multinomialnb_classifier = MultinomialNB()
print('cv f1 scores:',cross_val_score(multinomialnb_classifier,X, y,scoring='f1', cv=10)) # 10 folds cross validation
# Confusion Matrix Visualization
mnb_pred=cross_val_predict(multinomialnb_classifier, X, y,cv=10)
multinomialnb_classifier_cm=confusion_matrix(mnb_pred,y)
print('correct 0: {}, correct 1: {}, incorrect: {}'.format(multinomialnb_classifier_cm[0][0],multinomialnb_classifier_cm[1][1],multinomialnb_classifier_cm[1][0]+multinomialnb_classifier_cm[0][1]))
sns.heatmap(multinomialnb_classifier_cm, cmap='PuBu')

```

    cv f1 scores: [0.64210526 0.50097847 0.59803922 0.51968504 0.6541471  0.59322034
     0.65084746 0.5530303  0.74603175 0.75153374]
    correct 0: 3517, correct 1: 1861, incorrect: 2235





    <matplotlib.axes._subplots.AxesSubplot at 0x7efa9cdfc650>




![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_33_2.png)


#### ***Gaussian NB***


```python
#GaussianNB
gnb_classifier = GaussianNB()
X_gnb=X.toarray() #converting X to dense, required by GaussianNB
print('cv f1 scores:',cross_val_score(gnb_classifier,X_gnb, y,scoring='f1', cv=10))
# Confusion Matrix Visualization
gnb_pred=cross_val_predict(gnb_classifier, X_gnb, y,cv=10)
gnb_classifier_cm=confusion_matrix(gnb_pred,y)
print('correct 0: {}, correct 1: {}, incorrect: {}'.format(gnb_classifier_cm[0][0],gnb_classifier_cm[1][1],gnb_classifier_cm[1][0]+gnb_classifier_cm[0][1]))
sns.heatmap(gnb_classifier_cm, cmap='PuBu')

```

    cv f1 scores: [0.60815822 0.56313131 0.59114583 0.53026634 0.59338061 0.5819135
     0.60081191 0.58373206 0.5990566  0.56829268]
    correct 0: 1907, correct 1: 2341, incorrect: 3365





    <matplotlib.axes._subplots.AxesSubplot at 0x7efaa5733950>




![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_35_2.png)


For this dataset, multinomial NB has better accuracies than gaussian NB.

[back to top](#top)

<a id='lr'></a>
### Logistic Regression

#### ***Theory behind logistic regression***
Logistic regression uses the function $y=\frac{1}{1+e^{-(\beta_0+\beta_1X_1+\beta_2X_2...)}}$ in which $y$ is the class to be predicted,$X_n$ is the feature, and $\beta_n$ is the parameter to be paired with $X_n$. If the parameters and features are vectorized, the logistic regression function can be written as $y=\frac{1}{1+e^{-(\beta^{T}X)}}$ in which $\beta^{T}$ is the transpose of the parameter vector and X is the feature vector.

Logistic regression can be seen as linear regression which predicts continuous results transformed to predict discrete classes by applying the sigmoid function which follows $sigmoid(x)=\frac{1}{1+e^{-x}}$.

references
* [Logistic Regression Analysis](https://www.sciencedirect.com/topics/medicine-and-dentistry/logistic-regression-analysis)
* [Machine Learning Notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf)


#### ***Using logistic regression***

Sklearn's logistic regression module will be used here with 10 fold cross validation and f1 as the scoring metric.


```python
logisticreg_classifier = LogisticRegression()
print('cv f1 scores:',cross_val_score(logisticreg_classifier,X, y,scoring='f1', cv=10))
# Confusion Matrix Visualization
lr_pred=cross_val_predict(logisticreg_classifier, X, y,cv=10)
logisticreg_classifier_cm=confusion_matrix(lr_pred,y)
print('correct 0: {}, correct 1: {}, incorrect: {}'.format(logisticreg_classifier_cm[0][0],logisticreg_classifier_cm[1][1],logisticreg_classifier_cm[1][0]+logisticreg_classifier_cm[0][1]))
sns.heatmap(logisticreg_classifier_cm, cmap='PuBu')

# multinomialnb_classifier.fit(X_train, y_train)
# multinomialnb_classifier_pred = multinomialnb_classifier.predict(X_test)
```

    cv f1 scores: [0.61302682 0.51089109 0.50788091 0.51851852 0.58319039 0.54054054
     0.5503876  0.43076923 0.64171123 0.74209651]
    correct 0: 3721, correct 1: 1544, incorrect: 2348





    <matplotlib.axes._subplots.AxesSubplot at 0x7efa9d1926d0>




![png]({{ site.baseurl }}/images/disaster-tweet_files/disaster-tweet_38_2.png)


[back to top](#top)

<a id='LSTM'></a>
### LSTM

#### ***Theory behind LSTM***

LSTM (long short term memory) is a RNN (recurrent neural network) structure. RNNs are different from more traditional NNs because RNNs take into account of previous states, whcih makes them suitable for natural language processing as text have different positions. I tried to make a simplified rnn graph below. Because rnn receive both input from previous units and input from x, it will have a "memory" over time. 

```
->[rnn unit]->[rnn unit]->
       ^           ^
       |           |
       x           x
```

LSTM allows the architecture to detect associations between words over "extended time intervals"(Hochreiter & Schmidhuber, 1997)， and it solves the "vanishing gradient" problem in RNNs using gates. 

In LSTM's each unit, there are 3 gates: 
* forget gate: output 0/1 to decide if the previous state should be kept
* input gate: actives the input x to enter the unit, often using a sigmoid function
* output gate: decides how much the cell will output

reference:
* [Long Short-Term Memory](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


#### ***Using LSTM***

We will use keras to build LSTM. Before building the model.
1. Tokenize the text data and instead of using TF-IDF transformed data
    * Tokenizer(), transforms data into tokens and assign counts to tokens
2. Make an embed layer from the tokenized data
    * Embedding(len(tokenizer.word_index) + 1, 256,input_length = X.shape[1])
        * len(tokenizer.word_index) + 1 is number of words after tokenizing found [here](https://stackoverflow.com/questions/53525994/how-to-find-num-words-or-vocabulary-size-of-keras-tokenizer-when-one-is-not-as)
        * 256 is the embedding dimension
        * X.shape[1] is the size of input for each tweet
    * Adding Dropout to regularize the networks and prevent overfitting
    * Adding LSTM layers
    * Adding output dense layer


```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['cleaned_text'].values)
X = tokenizer.texts_to_sequences(df_train['cleaned_text'].values)
X = pad_sequences(X)


model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 128 ,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(256, return_sequences=True))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test,y_test))
```

    Model: "sequential_51"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_49 (Embedding)     (None, 23, 128)           2389760   
    _________________________________________________________________
    spatial_dropout1d_18 (Spatia (None, 23, 128)           0         
    _________________________________________________________________
    lstm_77 (LSTM)               (None, 23, 256)           394240    
    _________________________________________________________________
    dense_53 (Dense)             (None, 23, 1)             257       
    =================================================================
    Total params: 2,784,257
    Trainable params: 2,784,257
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/5
    215/215 [==============================] - 16s 74ms/step - loss: 0.6405 - accuracy: 0.6227 - val_loss: 0.6028 - val_accuracy: 0.6923
    Epoch 2/5
    215/215 [==============================] - 15s 68ms/step - loss: 0.5670 - accuracy: 0.6813 - val_loss: 0.5987 - val_accuracy: 0.6914
    Epoch 3/5
    215/215 [==============================] - 15s 71ms/step - loss: 0.5225 - accuracy: 0.7044 - val_loss: 0.6136 - val_accuracy: 0.6855
    Epoch 4/5
    215/215 [==============================] - 15s 68ms/step - loss: 0.4874 - accuracy: 0.7230 - val_loss: 0.6601 - val_accuracy: 0.6834
    Epoch 5/5
    215/215 [==============================] - 15s 68ms/step - loss: 0.4648 - accuracy: 0.7301 - val_loss: 0.6549 - val_accuracy: 0.6804





    <tensorflow.python.keras.callbacks.History at 0x7efa9cf8ff10>


