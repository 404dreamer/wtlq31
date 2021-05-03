import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from tensorflow.keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
import tensorflow_addons as tfa
from tf2crf import ModelWithCRFLoss, CRF

from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# loading training and testing data
data = pd.read_csv('../msra_train.csv') # training data
test_data = pd.read_csv('../msra_test.csv')# testing data

training_char = data.Character.unique()
testing_char = test_data.Character.unique()
illegal_chars = []
# removing all the Chinese characters which are in testing set but not in training set
for char in testing_char:
    if char not in training_char:
        illegal_chars.append(char)
        
test_data = test_data[~ test_data.Character.isin(illegal_chars)]

# loading radical dictionary (will use chise in the future)
df_radicals = pd.read_csv('../chise_radical.csv')                            
characters = df_radicals['character'].values
radicals = df_radicals['radical_info'].values
char_rad_dict = {}
for i in range (len(characters)):
    exec('char_rad_dict[characters[i]] =' +  radicals[i])

    
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(c, t) for c, t in zip(s["Character"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None
# training sentence
getter = SentenceGetter(data)
sentences = getter.sentences
# testing sentence
getter_te = SentenceGetter(test_data)
sentences_te = getter_te.sentences


df_char_token = pd.read_csv('../msra_char_token.csv')
df_radical_token = pd.read_csv('../msra_radical_token.csv')
df_tag_token = pd.read_csv('../msra_tag_token.csv')

char2idx = {}; radical2idx = {}; tag2idx = {}
for i in range (len(df_char_token)):
    char2idx[df_char_token.char[i]] = df_char_token.token[i]
idx2char = {i: c for c, i in char2idx.items()}

for i in range (len(df_radical_token)):
    radical2idx[df_radical_token.radical[i]] = df_radical_token.token[i]
idx2radical = {i : r for r, i in radical2idx.items()}

for i in range (len(df_tag_token)):
    tag2idx[df_tag_token.tag[i]] = df_tag_token.token[i]
idx2tag = {i: w for w, i in tag2idx.items()}


max_len_char = 90 # the length for each sentence (including character padding)
max_len_radical = 8 # the length for each character (including radical padding)

# character tokenrization
def char_token(sentences, char2idx = char2idx, max_len_char = max_len_char):
    X_char = [[char2idx[c[0]] for c in s] for s in sentences]
    X_char = pad_sequences(maxlen=max_len_char, sequences=X_char, value=char2idx["PAD"], padding='post', truncating='post')
    X_char = np.array(X_char)
    return X_char

X_char = char_token(sentences)
X_char_te = char_token(sentences_te)


# radical tokenrization
def radical_token(sentences, max_len_char = max_len_char, max_len_radical = max_len_radical,
                 radical2idx = radical2idx, char_rad_dict = char_rad_dict):
    X_radical = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len_char):
            word_seq = []
            for j in range(max_len_radical): 
                try:
                    char = sentence[i][0][j]
                    if char in char_rad_dict.keys():     
                        radicals = char_rad_dict[char]
                        for i in range(len(radicals)):
                            if i < max_len_radical - 1:    
                                word_seq.append(radical2idx.get(radicals[i]))
                    else:
                        word_seq.append(radical2idx.get("UNK"))
                except:
                    if len(word_seq) < max_len_radical:
                        word_seq.append(radical2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_radical.append((sent_seq))
    X_radical = np.array((X_radical))
    return X_radical

X_radical = radical_token(sentences)
X_radical_te = radical_token(sentences_te)


def tag_token(sentences, max_len_char = max_len_char, tag2idx = tag2idx):
    y = [[tag2idx[c[1]] for c in s] for s in sentences]
    y = pad_sequences(maxlen=max_len_char, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')
    return y

y = tag_token(sentences)
y_te = tag_token(sentences_te)

### Deep learning model

n_chars = len(df_char_token)
n_radical = len(df_radical_token)
n_tags = len(df_tag_token)

num_epoch = 200
lr_rate = 0.002
char_dim = 125
radical_dim = 0
batch_size = 256


# input and embeddings for characters
char_in = Input(shape=(max_len_char,), )
emb_char = Embedding(input_dim = n_chars, output_dim = char_dim, input_length = max_len_char, mask_zero = True)(char_in)

# main LSTM
main_lstm = Bidirectional(LSTM(units=char_dim + radical_dim, return_sequences=True,dropout = 0.5, recurrent_dropout=0.5))(emb_char)
dense = TimeDistributed(Dense(char_dim + radical_dim, activation = None))(main_lstm)
crf = tfa.layers.CRF(n_tags)
out = crf(dense)
model = Model(inputs =char_in, outputs = out)
model = ModelWithCRFLoss(model)
opt = tf.keras.optimizers.Adam(learning_rate=lr_rate)
model.compile(optimizer=opt)

dir_path = 'msra_scheme_3'
try:
    os.mkdir(dir_path)
except:
    pass

checkpoint_path = dir_path + "/cp-{epoch:04d}.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)

model.fit(X_char,y,batch_size= batch_size, epochs=num_epoch, validation_split=0.1, verbose=1,callbacks=[cp_callback])

# Evaluation Process
F1_scores = {}
# filtering the padding tokens
y_te_mask = y_te != 0
y_true = y_te[y_te_mask]
# recove tokens to tags
y_true_recovered = [[idx2tag[token] for token in y_true]]

for n in range (1, num_epoch + 1):
    weight_path = dir_path + "/cp-{:04d}.ckpt".format(n)
    model.load_weights(weight_path)
    
    y_pred_result = model.predict(X_char_te)[0]
    y_pred = y_pred_result[y_te_mask]
    # recove tokens to tags
    y_pred_recovered = [[idx2tag[token] for token in y_pred]]
    
    f1 = f1_score(y_true_recovered, y_pred_recovered)
    F1_scores[n] = f1
    
df_result = pd.DataFrame(data = {'epoch': F1_scores.keys(), 'F1':F1_scores.values()})
optimal_epoch = df_result[df_result.F1 == df_result.F1.max()]['epoch'].values[0]
df_result.to_csv('msra_scheme_3_result.csv')

optimal_path = dir_path + "/cp-{:04d}.ckpt".format(optimal_epoch)
model.load_weights(optimal_path)
y_pred_result = model.predict(X_char_te)[0]
y_pred = y_pred_result[y_te_mask]
y_pred_recovered = [[idx2tag[token] for token in y_pred]]
print('Optimal F1 Score: ',f1_score(y_true_recovered, y_pred_recovered))
print(classification_report(y_true_recovered, y_pred_recovered))
