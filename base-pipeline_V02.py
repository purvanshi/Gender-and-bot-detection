#%%
########## Load Data ##########
import pandas as pd
import numpy as np
from collections import Counter

df_train = pd.read_csv("./input/train.csv")
df_test = pd.read_csv("./input/test.csv")

print("Train shape : ", df_train.shape)
print("Test shape : ", df_test.shape)


#%%
########## Quick test for null/blanks ##########

isNullTrain = df_train.isnull().sum()
isNullTest = df_test.isnull().sum()

if isNullTrain['qid'] != 0 or isNullTrain['question_text'] != 0 or isNullTrain['target'] != 0:
    print("df_train contains null segments!")
if isNullTest['qid'] != 0 or isNullTest['question_text'] != 0:
    print("df_test contains null segments!")

def get_blanks(df):
    ''' Returns a list containing the indices of every "" or space found in df["question_text"] '''
    blanks = []
    for i, _, question_text, _ in df_train.itertuples():
        if question_text.isspace() or question_text == "" or len(question_text) == 0 or question_text == "  " or question_text == None:
            blanks.append(i)
    return blanks

print("Number of blanks/spaces in train: ", len(get_blanks(df_train)),
    "\nNumber of blanks/spaces in test: ", len(get_blanks(df_test)))


#%%
########## Initial preprocessing ##########

# Create lowercase question set
df_train['question_text'] = df_train['question_text'].str.lower()
df_test['question_text'] = df_test['question_text'].str.lower()

# Fill missing values
df_train.fillna("_##_", inplace=True)  
df_test.fillna("_##_", inplace=True)

print("Preprocessing done...")


#%%
########## Create counters & show statistics ##########

def split_in_words(data):
    return data.split()

def split_data(df):
    ''' Split data into arrays for questions, all_words, insincere_words and sincere_words '''
    questions = []
    all_words = []
    insincere_words = []
    sincere_words = []

    for i, qid, question_text, target in df.itertuples():
        questions.append(split_in_words(question_text))
        for w in split_in_words(question_text):
            all_words.append(w)
            if target == 1:
                insincere_words.append(w)
            else:
                sincere_words.append(w)

    return questions, all_words, insincere_words, sincere_words

def word_count(words):
    return Counter(words)

# Get all words, words found in insincere questions and words found in sincere questions
questions, all_words, insincere_words, sincere_words = split_data(df_train)

# Create counters
all_words_count = word_count(all_words)
insincere_words_count = word_count(insincere_words)
sincere_words_count = word_count(sincere_words)

# Get length of longest question
question_lengths = Counter([len(x) for x in questions])
max_question_length = max(question_lengths)
average_question_length = int(sum(question_lengths) / len(question_lengths))

print("Total number of words: ", len(all_words),
    "\nNumber of unique words: ", len(set(all_words)),
    "\nNumber of words in insincere questions: " , len(insincere_words),
    "\nNumber of unique words in insincere questions: ", len(set(insincere_words)),
    "\nNumber of words in sincere questions: ", len(sincere_words),
    "\nNumber of unique words in sincere questions: ", len(set(sincere_words)),
    "\nMax question length: ", max_question_length,
    "\nAverage question length: ", average_question_length,
    "\nNumber of sincere questions: ", df_train['target'].value_counts()[0],
    "\nNumber of insincere questions: ", df_train['target'].value_counts()[1])

print("\nMost common words: " , all_words_count.most_common(50),
    "\n\nMost common insincere words: ", insincere_words_count.most_common(50),
    "\n\nMost common sincere words: " , sincere_words_count.most_common(50))

print("\n\n\n")
print("Least common words")
print(all_words_count.most_common()[-50:-1])

print("\n\n\n")
print("Most common question lengths:")
print(question_lengths.most_common(50))

'''
A good idea would be to plot the following:
Most common words
Most common insincere words
Most common sincere words
Number of insincere examples
Number of sincere examples
Longest question
Average length of questions

print words only found in insincere questions
print words only found in sincere questions
'''

#%%
########## Build Vocabulary ##########

def split_questions(questions):
    questions_split = []
    for question in questions:
        questions_split.append(split_in_words(question))
    return questions_split

def vocabulary(counts):
    ''' Defines a vocabulary for the words in counts '''
    return sorted(counts, key=counts.get, reverse=True)

def vocabulary_to_integer(vocab):
    ''' Map each vocab words to an integer.
        Starts at 1 since 0 will be used for padding.'''
    return {word: ii for ii, word in enumerate(vocab, 1)}

def integer_to_vocabulary(vocab_to_int):
    ''' Take a vocab_to_int and flip it so we can map an integer to a word '''
    int_to_vocab = {}
    for key, val in vocab_to_int.items():
        int_to_vocab[val] = key
    return int_to_vocab

def strings_to_integers(questions_str, vocab_to_int):
    ''' Converts the question strings into integers '''
    questions = []
    for q in questions_str:
        q_tmp = []
        for w in q:
            q_tmp.append(vocab_to_int[w])
        questions.append(q_tmp)
    return questions

# Create vocab
vocab = vocabulary(all_words_count)
vocab_to_int = vocabulary_to_integer(vocab)
int_to_vocab = integer_to_vocabulary(vocab_to_int)


#%% ########## Tokenize (no special embedding) ##########

questions_split = split_questions(df_train['question_text'])
questions_int = strings_to_integers(questions_split, vocab_to_int)
labels = df_train['target']


#%%
########## Load embeddings ##########

def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    
    if file.split('/')[-1] == 'wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o) > 100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index

# Only use GloVe for now to save memory!
glove = './input/embeddings/glove.840B.300d/glove.840B.300d.txt'
# paragram =  './input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
# wiki_news = './input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
# google_news = './input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

print("Extracting GloVe embedding")
embed_glove = load_embed(glove)
# print("Extracting Paragram embedding")
# embed_paragram = load_embed(paragram)
# print("Extracting FastText embedding")
# embed_fasttext = load_embed(wiki_news)
# print("Extracting Google embedding")
# embed_google = load_embed(wiki_news)
print("Done extracting!")


#%%
########## Create vocab & check coverage ##########
import operator

def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embeddings_index):
    ''' Checks the coverate of a vocabulary in a given embedding.
        Returns an array of out of vocab words (oov) '''
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words


vocab_original = build_vocab(df_train['question_text'])
# vocab_glove = build_vocab(df_train['cleaned_questions'])

print("Glove (original coverage) : ")
oov_glove = check_coverage(vocab_original, embed_glove)


#%%
########## Preprocessing for increased Glove coverage ##########

contraction_mapping = {
    "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", 
    "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", 
    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
    "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
    "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", 
    "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", 
    "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
    "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
    "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 
    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", 
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", 
    "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", 
    "that'd've": "that would have", "that's": "that is", "there'd": "there would", 
    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", 
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", 
    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 
    "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", 
    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", 
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
    "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 
    "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 
    "you're": "you are", "you've": "you have" 
    }

def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")

def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

print("- Known Contractions -")
print("   Glove :")
print(known_contractions(embed_glove))

# Add lowercase to embedding
add_lower(embed_glove, vocab_original)

# Remove contractions
df_train['question_text_glove'] = df_train['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

vocab_glove = build_vocab(df_train['question_text_glove'])
print("Glove : ")
oov_glove = check_coverage(vocab_glove, embed_glove)

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {
    "‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", 
    "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', "£": "e", 
    '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', 
    '∅': '', '³': '3', 'π': 'pi'
    }

print("Glove: (Unknown punctuations)")
print(unknown_punct(embed_glove, punct))

df_train['question_text_glove'] = df_train['question_text_glove'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
vocab_glove = build_vocab(df_train['question_text_glove'])

print("Glove: (Improved coverage)")
oov_glove = check_coverage(vocab_glove, embed_glove)

mispell_dict = {
    'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 
    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 
    'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 
    'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 
    'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 
    'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 
    'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 
    'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', 
    '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 
    'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 
    'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'
    }

df_train['question_text_glove'] = df_train['question_text_glove'].apply(lambda x: correct_spelling(x, mispell_dict))

vocab_glove = build_vocab(df_train['question_text_glove'])
print("Glove: (final coverage)")
oov_glove = check_coverage(vocab_glove, embed_glove)

print(vocab_glove[:10])


#%%
########## Create train/validation data ##########

from sklearn.utils import shuffle

TRAIN_FRACTION = 0.8

def pad_features(questions, sequence_length=50):
    ''' Pad each question with zeros to the same length '''
    features = np.zeros((len(questions), sequence_length), dtype=int)
    for i, row in enumerate(questions):
        features[i, -len(row):] = np.array(row)[:sequence_length]
    return features

def split_into_train_test(questions, labels, train_fraction=0.8):
    ''' Create valid features, labels that we can later use in our models '''

    questions, labels = shuffle(questions, labels)

    # Decide on number of sample for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(questions[:train_end])
    valid_features = np.array(questions[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    X_train, X_valid = np.array(train_features), np.array(valid_features)
    Y_train, Y_valid = np.array(train_labels), np.array(valid_labels)
    
    return X_train, X_valid, Y_train, Y_valid


# Get length of longest question
question_lengths = Counter([len(x) for x in questions_int])
sequence_length = max(question_lengths)

# Pad all questions to the length of the longest question
features = pad_features(questions_int, sequence_length=sequence_length)

## test statements - do not change - ##
assert len(features)==len(questions_int), "Your features should have as many rows as questions."
assert len(features[0])==sequence_length, "Each feature row should contain seq_length values."

X_train, X_valid, Y_train, Y_valid = split_into_train_test(features, labels, train_fraction=TRAIN_FRACTION)

assert len(X_train)==len(Y_train), "Number of features must be the same a number of labels"
assert len(X_valid)==len(Y_valid), "Number of features must be the same a number of labels"

# Compare original vs padded
print("Original vs padded")
print(questions_int[25])
print(features[25])

print("X_train: ", X_train.shape)
print("Y_train: ", Y_train.shape)
print("X_valid: ", X_valid.shape)
print("Y_valid: ", Y_valid.shape)


#%%
########## Some manual verification of data ##########

def question_int_to_string(question, int_to_vocab):
    s = ""
    for w in question:
        if w != 0:
            s += int_to_vocab[w] + " "
    return s

# Convert a question from features back to a string and compare with original
print(question_int_to_string(features[15], int_to_vocab))
print(" ".join(questions_split[15]))


#%%
########## Create subset to run initial classifiers on ##########

def create_subset(train_subset_size, validation_subset_size, seq_len, X_train, Y_train, X_valid, Y_valid):
    ''' Split train and validation set into a subset. Used in initial testing '''
    X_train_sub = np.zeros((train_subset_size, seq_len), dtype=int)
    Y_train_sub = np.zeros((train_subset_size, 1), dtype=int)
    X_valid_sub = np.zeros((validation_subset_size, seq_len), dtype=int)
    Y_valid_sub = np.zeros((validation_subset_size, 1), dtype=int)

    for i in range(0, train_subset_size):
        X_train_sub[i] = X_train[i]
        # if Y_train[i] == 1:
        #     Y_train_sub[i] = [0, 1]
        # else:
        #     Y_train_sub[i] = [1, 0]
        Y_train_sub[i] = Y_train[i]

    for i in range(0, validation_subset_size):
        X_valid_sub[i] = X_valid[i]
        Y_valid_sub[i] = Y_valid[i]

    print("X_train_sub: ", X_train_sub.shape)
    print("Y_train_sub: ", Y_train_sub.shape)
    print("X_valid_sub: ", X_valid_sub.shape)
    print("Y_valid_sub: ", Y_valid_sub.shape)
    # print(Y_train_sub[:100])

    return X_train_sub, Y_train_sub, X_valid_sub, Y_valid_sub

X_train_sub, Y_train_sub, X_valid_sub, Y_valid_sub = create_subset(1000, 1000, 134, X_train, Y_train, X_valid, Y_valid)


#%% ########## Run simple classifier ##########

from time import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svclassifier = SVC(kernel='linear', C=0.5)

print(len(X_train_sub))
print(len(X_valid_sub))

# Train and predict
t0 = time()
svclassifier.fit(X_train_sub, Y_train_sub)
train_time = time() - t0
t0 = time()
Y_pred = svclassifier.predict(X_valid_sub) 
pred_time = time() - t0

print("Training time: %0.3fs" % train_time,
    "\nPrediction time: %0.3fs" % pred_time)

print("\nConfusion matrix:\n", confusion_matrix(Y_valid_sub, Y_pred))
print("\nReport:\n", classification_report(Y_valid_sub, Y_pred))


#%%
from sklearn.metrics import f1_score

print("\nReport:\n", classification_report(Y_valid_sub, Y_pred))
print("Binary: ", f1_score(Y_valid_sub, Y_pred, average='binary'))
print("Micro: ", f1_score(Y_valid_sub, Y_pred, average='micro'))
print("Macro: ", f1_score(Y_valid_sub, Y_pred, average='macro'))
print("Weighted: ", f1_score(Y_valid_sub, Y_pred, average='weighted'))

#%%
########## Can I get this to work with pytorch? ##########
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import copy

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Create torchvision dataset
tensor_x_t = torch.stack([torch.Tensor(i) for i in X_train_sub]) # transform to torch tensors
tensor_y_t = torch.stack([torch.Tensor(i).long() for i in Y_train_sub])
tensor_x_v = torch.stack([torch.Tensor(i) for i in X_valid_sub])
tensor_y_v = torch.stack([torch.Tensor(i).long() for i in Y_valid_sub])

print("tensor_x_t: ", tensor_x_t.shape)
print("tensor_y_t: ", tensor_y_t.shape)
print("tensor_x_v: ", tensor_x_v.shape)
print("tensor_y_v: ", tensor_y_v.shape)

torch_train = torch.utils.data.TensorDataset(tensor_x_t, tensor_y_t)
torch_val = torch.utils.data.TensorDataset(tensor_x_v, tensor_y_v)

torch_train_loader = torch.utils.data.DataLoader(dataset=torch_train,
                                                batch_size=8,
                                                shuffle=True)

torch_val_loader = torch.utils.data.DataLoader(dataset=torch_val,
                                                batch_size=8,
                                                shuffle=False)
# Save loaders in single dict
dataloaders_dict = {"train": torch_train_loader, "val": torch_val_loader}

# Try simple model
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 150)
        self.fc2 = nn.Linear(150, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Check for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Running on device: ", device)
    print("torch.cuda.current_device()", torch.cuda.current_device())
    print("torch.cuda.device(0)", torch.cuda.device(0))
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))

# Create model
model = Net(134, 2)

# Print the model we just instantiated
print(model)

# Send model to deivce (GPU)
model = model.to(device)

# Define loss func
loss_function = nn.CrossEntropyLoss()

# We use the Adam optimizer algorithm
# https://pytorch.org/docs/stable/optim.html
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#%%
# Train the model
def train_and_val_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    ''' Train the model and then load the weights that gave the best validation results '''

    print("Training on device: {}".format(device))

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        time_elapsed = time.time() - since
        print('Epoch {}/{} [Duration: {:.0f}m {:.0f}s]'.format(epoch, num_epochs - 1, time_elapsed // 60, time_elapsed % 60))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                labels = labels.squeeze_() # Change labels from size [8, 1] to [8]
                
                # debug print
                # print("inputs.shape: {}".format(inputs.shape))
                # print("labels.shape: {}".format(labels.shape))
                # print("labels: {}".format(labels))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)

                    # print("outputs.shape: {}".format(outputs.shape))
                    # print("outputs: {}".format(outputs))

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print('Best val Acc: {:4f} in Epoch: {:.0f}'.format(best_acc, best_epoch))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} in Epoch: {:.0f}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataloaders, classes, execution_number, total_runs):
    ''' Test the model on some test set '''
    
    print("Testing on device: {}".format(device))
    with torch.no_grad():
        since = time.time()

        # Vars for total accuracy
        correct = 0
        total = 0
        # Vars for accuracy per class
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        for inputs, labels in dataloaders['test']:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Overall accuracy
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # Accuracy per class
                c = (preds == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                    

        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s [Run: {}/{}]'.format(time_elapsed // 60, time_elapsed % 60, execution_number, total_runs))
        print('Accuracy of the network on the {} test images: {:.4f} %'.format(total, 100.0 * correct / total))
            
        for i in range(10):
            print('Accuracy of {} : {:.4f} %'.format(classes[i], 100.0 * class_correct[i] / class_total[i]))        
        
        print()

        return correct, total, class_correct, class_total

#
# Run training
#
model, hist = train_and_val_model(model, dataloaders_dict, loss_function, optimizer, num_epochs=5)

