#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install pyphen')
# get_ipython().system('pip install nltk')
# get_ipython().system('pip install emoji')
# get_ipython().system('pip install -U spacy')


# In[2]:


import os
import pandas as pd
import codecs
import re
import xml.etree.ElementTree as ET
import numpy as np
import shutil
from shutil import copyfile
import emoji
import re
import operator
import argparse
from collections import Counter, OrderedDict, defaultdict


# In[3]:


# get_ipython().system('python -m spacy download en_core_web_md')


# In[27]:


import pyphen
PYPHEN_DIC = pyphen.Pyphen(lang='en')
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
import spacy
spacy_nlp = spacy.load('en_core_web_md')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer


# ### Character flooding
# eg looooove, floooood
# Find words that have consecutive characters appearing in them more than 2 times

# In[6]:

parser = argparse.ArgumentParser(description='Test Data Preparation and Output')
parser.add_argument('--input_dir', type=str, required=True, help="Input directory for test data")
parser.add_argument('--output_dir', type=str, default='en_features', help="Output folder")

global opt
opt = parser.parse_args()
print(opt)

def CharacterFlooding(tweet):
    tknzr = TweetTokenizer()
    tweet_words = tknzr.tokenize(tweet)
    floodings = 0
    regexpURL = re.compile(r'https?:\/\/.[^\s]*|www\.[^\s]*')
    for word in tweet_words:
        if not regexpURL.search(word) and len(re.findall(r'(\w)\1{2,}',word)) > 0: 
            floodings += 1
    return floodings


# ### Counting Capital Letters

# In[7]:


def NumberOfCapitalLetters(tweet):
    now, words = NumberOfWords(tweet)
    nocl = 0
    for word in words:
        nocl += sum(1 for c in word if c.isupper())
    avg_nocl = nocl / now
    return nocl, avg_nocl


# ### Counting Emoticons

# In[8]:


def NumberOfEmoticons(tweet):
    emojis = [i for i in tweet if i in emoji.UNICODE_EMOJI]
    return len(emojis)


# ### Exploring Topics

# In[9]:


import pickle
SEMCAT_word_concept_dict = pickle.load(open('./pan19-author-profiling-training/kb/SEMCAT2018_word_concept_dict.p', 'rb'), encoding='utf-8')
semcor_word_concept_dict = pickle.load(open('./pan19-author-profiling-training/kb/semcor_noun_verb.supersenses.en_word_concept_dict.p', 'rb'), encoding='utf-8')
SEMCAT_concepts = set([item for sublist in SEMCAT_word_concept_dict.values() for item in sublist])
semcor_concepts = set([item for sublist in semcor_word_concept_dict.values() for item in sublist])
# merge
wordTopicsDict = SEMCAT_word_concept_dict
for word, concept_set in semcor_word_concept_dict.items():
    wordTopicsDict[word] = wordTopicsDict[word].union(concept_set)
topics = SEMCAT_concepts.union(semcor_concepts)
topicNames = sorted(list(topics), key=lambda t: t)
print(wordTopicsDict['car'], SEMCAT_word_concept_dict['car'], semcor_word_concept_dict['car'])
print("Number of topics: ", len(topics))
print(topicNames)


# In[10]:


def Topics(tweet):
    topicsFrequency = {t:0 for t in topics}
    tweet = spacy_nlp(tweet)
    lemmas = [token.lemma_ for token in tweet]
    for token in tweet:
        lemma = token.lemma_
        topicSet = wordTopicsDict.get(lemma, set())
        for t in topicSet:
            topicsFrequency[t] += 1
    topicsFrequency = OrderedDict(sorted(topicsFrequency.items(), key=lambda e: e[0]))
    topicsValues = list(topicsFrequency.values())
    return topicsValues


# ### Counting URLs

# In[11]:


def NumberOfURLs(tweet):
    tknzr = TweetTokenizer()
    tweet_words = tknzr.tokenize(tweet)
    regexpURL = re.compile(r'https?:\/\/.[^\s]*|www\.[^\s]*')
    urlNumber = 0
    for word in tweet_words:
        if regexpURL.search(word): 
            urlNumber += 1
    return urlNumber


# ### Counting Repeated Words
# Compute:
# - number of tokens repeated more (or equal) than *k* times 
# - maximum number of repetition (of a single token) (>=*k*)
# 
# For example (k=3) in the tweet: *"Hairy cats like other cats that are not hairy. However, hairy dogs like cats that are not hairy."*
# - the tokens repeated more (or equal) than 3 times are *hairy* and *cats*, so the number of tokens repeated more than 3 times is 2
# - the token *hairy* is repeated most of the time: 4 times

# In[12]:


def wordsRepeated(tweet, k=3):

    frequency = defaultdict(int)
    tweet = spacy_nlp(tweet)
    for token in tweet:
        if token.lemma_ not in spacy_stopwords:
            frequency[token.lemma_] += 1
    maxAppearance = 0
    numberOfTokensRepeated = 0
    if len(frequency.items()) > 0:
        maxAppearance = max(frequency.items(), key=operator.itemgetter(1))[1]
        numberOfTokensRepeated = sum(1 for (lemma, freq) in frequency.items() if freq >= k)
    return maxAppearance, numberOfTokensRepeated


# ### Number of POS tags

# In[13]:


all_pos_tags = ["NO_TAG", "ADJ", "ADP", "ADV","AUX", "CONJ","CCONJ","DET",
                      "INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM",
                      "VERB","X","EOL","SPACE"]


# In[14]:


def POSTags(tweet):
    tweet = spacy_nlp(tweet)
#     pos_list = []
    c = Counter()
    c.update({x:0 for x in all_pos_tags})
    pos_list = [token.pos_ for token in tweet]
    assert len(set(pos_list).difference(set(all_pos_tags))) == 0
    c.update(pos_list)
    c = OrderedDict(sorted(c.items(), key=lambda e: e[0]))
    return list(c.values())


# ### Number of Sentences

# In[15]:


def NumberOfSentences(tweet):
    sentences = sent_tokenize(tweet)
    return len(sentences)


# ### Number of Words
# using nltk's TweetTokenizer

# In[16]:


def NumberOfWords(tweet):
    tknzr = TweetTokenizer()
#     words = word_tokenize(tweet)
    tweet_words = tknzr.tokenize(tweet)
    return len(tweet_words), tweet_words


# ### Readability Score
# using Flesch–Kincaid readability tests (higher score means the tweet is more easy to read)

# In[17]:


def Readability(tweet):
    totalSentences = float(NumberOfSentences(tweet))
    totalTweetWords, tweetWords = NumberOfWords(tweet)
    totalTweetWords = float(totalTweetWords)
    regexpURL = re.compile(r'https?:\/\/.[^\s]*|www\.[^\s]*')
    totalSyllables = 0.0
    for word in tweetWords:
        if not regexpURL.search(word):
            hyphenated = PYPHEN_DIC.inserted(word)
            syllables = hyphenated.count("-") + 1 - hyphenated.count("--")
            totalSyllables += syllables
#         else:
#             totalTweetWords -= 1.0
    if totalSentences > 0 and totalTweetWords > 0:
        score = 206.835 - 1.015 * (totalTweetWords/totalSentences) - 84.6 * (totalSyllables/totalTweetWords)
    else:
        print("Readability issue")
        score = 0.0
    return score


# ### Feature Names

# In[18]:


fNames = ['nos', 'now', 'readabilityScore', 'noclPerWord', 'noe', 'nocf', 'noURL', 'maxWordAppearancePerTweet', 'nowr']
fNames.extend(all_pos_tags)
fNames.extend(topicNames)
fNames = ['avg_'+f for f in fNames]
fNames.append('maxWordAppearancePerProfile')
fNames = sorted(fNames)
print(len(fNames))
print(fNames)
fNames_dir = './pan19-author-profiling-training/en_features/'
if not os.path.exists(fNames_dir):
    os.makedirs(fNames_dir)
pickle.dump(fNames, open(fNames_dir+'feature_names.p', 'wb'))


# ### Compute features for each tweet
# ngrams are not used

# In[19]:


def GetFeatures(tweet):
    features = {}
    features['nos'] = NumberOfSentences(tweet)
    features['now'], words = NumberOfWords(tweet)
    features['readabilityScore'] = Readability(tweet)
    nocl, features['noclPerWord'] = NumberOfCapitalLetters(tweet)
    features['noe'] = NumberOfEmoticons(tweet)
    features['nocf'] = CharacterFlooding(tweet)
    features['noURL'] = NumberOfURLs(tweet)
    features['maxWordAppearancePerTweet'], features['nowr'] = wordsRepeated(tweet)
#     ngram = nGram(tweet, ngram_model)
    pos = POSTags(tweet)
    posTags = sorted(all_pos_tags)
    for i in range(len(posTags)):
        features[posTags[i]] = pos[i]
    topicValues = Topics(tweet)
    for i in range(len(topicNames)):
        features[topicNames[i]] = topicValues[i]
#     features = [nos, now, readabilityScore, avg_nocl, noe, nocf]
#     ngramFeatures = nGram(tweet, ngram_model)
#     for i in range(ngramFeatures.shape[1]):
#         name = ngram_features[i]
#         feat = ngramFeatures[0,i]
    return features


# ### Max number of word repetitions for a profile

# In[20]:


def maxWordRepetitionsPerProfile(TweetGenerator):
    tknzr = TweetTokenizer()
    frequency = defaultdict(int)
    for tweet in TweetGenerator:
        tweet = tweet['data']
        tweet = spacy_nlp(tweet)
        for token in tweet:
            if token.lemma_ not in spacy_stopwords:
                frequency[token.lemma_] += 1
    maxAppearance = 0
    if len(frequency.items()) > 0:
        maxAppearance = max(frequency.items(), key=operator.itemgetter(1))[1]
    return maxAppearance


# In[21]:


def additionalProfileFeatures(TweetGenerator, features, featureNames):
    maxWordAppearance = maxWordRepetitionsPerProfile(TweetGenerator)
    features.append(maxWordAppearance)
    featureNames.append('maxWordAppearancePerProfile')
    return features, featureNames


# ### Compute average of features for each Twitter profile

# In[22]:


def cummulateFeatures(profileFeatures):
    denom = len(profileFeatures)
    cummulatedFeatures = defaultdict(float)
    for tweetFeature in profileFeatures:
        for featureName in tweetFeature.keys():
            cummulatedFeatures[('avg_'+featureName)] += tweetFeature[featureName]
    for k, v in cummulatedFeatures.items():
        cummulatedFeatures[k] = cummulatedFeatures[k]/denom
    cummulatedFeatures = OrderedDict(sorted(cummulatedFeatures.items(), key=lambda e: e[0]))
    featureNames = list(cummulatedFeatures.keys())
    features = list(cummulatedFeatures.values())
    return features, featureNames


# In[23]:


def writeProfileFeatures(directory, profileID, profileFeatures, Truth, TweetGenerator):
    profileName = Truth[0][profileID]
#     mode = roleDict[profileName]
    features, featureNames = cummulateFeatures(profileFeatures)
    features, featureNames = additionalProfileFeatures(TweetGenerator,  features, featureNames)
    features = [str(f) for f in features]
    features = '\t'.join(features)
    with codecs.open(directory+'/features.txt', "a", "utf-8-sig") as text_file:
        text_file.write(profileName+'\t'+features+'\t'+Truth[1][profileID]+'\t'+Truth[2][profileID]+'\n')


# In[24]:


def LoadProfile(input_file):    
    Profile = ET.parse(input_file)
    ProfileRoot = Profile.getroot()
    Profile_attr = ProfileRoot.attrib
    for tweet in Profile.iter('document'):        
        tweet_dict = Profile_attr.copy()
        tweet_dict.update(tweet.attrib)
        tweet_dict['data'] = tweet.text
        yield tweet_dict


# In[25]:


def ProcessFolder(root_directory, input_directory, output_directory):    
    #Create output directory (if it does not exist yet)
    profile_directory = root_directory+'/'+output_directory+'/'+'profile'
    tweet_directory = root_directory+'/'+output_directory+'/'+'tweet'
    if os.path.exists(root_directory+'/'+output_directory):
        shutil.rmtree(root_directory+'/'+output_directory)
    os.mkdir(root_directory+'/'+output_directory)
    os.mkdir(profile_directory)
    os.mkdir(tweet_directory)
    #Read labels (and file names)
    Truth = pd.read_csv(root_directory+'/'+input_directory+'/truth.txt', sep=":::", header=None, engine='python')
    #Iterate over all user names, and process the corresponding file names
    for i in range(0,Truth.shape[0]):
#     for i in range(0,2):        
        #Open text file for output   
        print(Truth[0][i])
        profileFeatures = []
        TweetGenerator = LoadProfile(root_directory+'/'+input_directory+'/'+Truth[0][i]+'.xml')
        with codecs.open(tweet_directory+'/'+Truth[0][i]+'.txt', "w", "utf-8-sig") as text_file: 
            for tweet in TweetGenerator:
                tweetFeatures = GetFeatures(tweet['data'])
                profileFeatures.append(tweetFeatures)
                tweetFeatures = [str(f) for f in tweetFeatures.values()]
                tweetFeatures = '\t'.join(tweetFeatures)
                text_file.write(tweetFeatures + '\n')   
        TweetGenerator = LoadProfile(root_directory+'/'+input_directory+'/'+Truth[0][i]+'.xml')
        writeProfileFeatures(profile_directory, i, profileFeatures, Truth, TweetGenerator)


# In[ ]:


ProcessFolder('./', opt.input_dir, opt.output_dir)


# In[ ]:




