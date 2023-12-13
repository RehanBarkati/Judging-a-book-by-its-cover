import numpy as np
import pandas as pd
import torchtext  # 0.9.1
import torch   # 1.8.1+cpu
import spacy   #3.4.3
import torch.nn as nn
import torch.nn.functional as F
import random
import re , string
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchtext.vocab import GloVe
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#this will be used for tokenizing 
tok = spacy.load("en_core_web_sm")

X_train=pd.read_csv(sys.argv[1]+'\\train_x.csv')[['Title']]
y_train=pd.read_csv(sys.argv[1]+'\\train_y.csv')['Genre']

X_test=pd.read_csv(sys.argv[1]+'\\non_comp_test_x.csv')[['Title']]
y_test=pd.read_csv(sys.argv[1]+'\\non_comp_test_y.csv')['Genre']

def normalize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    text = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(text)]

#count number of occurences of each word
frequency = Counter()
for index, row in X_train.iterrows():
    frequency.update(normalize(row['Title']))
    
#creating mapping
vocab_index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in frequency:
    vocab_index[word] = len(words)
    words.append(word)



def words_to_numbers(text, vocab_index, N=60):
    text= normalize(text)
    enc = np.zeros(N, dtype=int)
    i=0
    for word in text:
        if word in vocab_index:
            enc[i]=vocab_index[word]
        else:
            enc[i]=vocab_index["UNK"]
        i=i+1
        if i>=N:
            break
    
    return enc,i


def preprocess(data,vocab_index):
    data['feature']= data['Title'].apply(lambda x: np.array(words_to_numbers(x,vocab_index )))
    data.drop('Title',axis=1,inplace=True)
    return data


preprocess(X_train,vocab_index)
preprocess(X_test,vocab_index)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx],self.X[idx][1],idx

X_train=list(X_train['feature'])
y_train=list(y_train)
X_test=list(X_test['feature'])
y_test=list(y_test)

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

batch_size = 100
vocab_size = len(words)
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

embedded_globe = GloVe(name='6B',dim=300)
dic = embedded_globe.stoi
vector= embedded_globe.vectors


def embedding_matrix(dic,vector, word_freq, emb_size = 300):
    vocab_size = len(word_freq) + 2
    vocab_idx = {}
    vocab = ["", "UNK"]
    Weight_matrix = np.zeros((vocab_size, emb_size), dtype="float32")
    Weight_matrix[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
    Weight_matrix[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
    vocab_idx["UNK"] = 1
    i = 2
    for word in word_freq:
        if word in dic:
            Weight_matrix[i] = vector[dic[word]]
        else:
            Weight_matrix[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_idx[word] = i
        vocab.append(word)
        i += 1   
    return Weight_matrix

embeddings =embedding_matrix(dic,vector,frequency)


torch.manual_seed(0)
class RNN(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim,weights) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(weights))
        self.embeddings.weight.requires_grad = False ## freeze embeddings
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True,bidirectional=True)
        self.linear1 =nn.Linear(2*hidden_dim, hidden_dim)
        self.linear2=nn.Linear(hidden_dim, 30)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x,s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        _,h = self.rnn(x)
#         print(h.shape),out.shape torch.Size([2, 100, 128]) ,torch.Size([100, 60, 256])
        h = torch.cat((h[-2,:,:], h[-1,:,:]),dim=1)
        output= torch.tanh(self.linear1(h))
        # output= torch.tanh(self.linear1(h[-1]))
        output=self.linear2(output)
        return output
      
def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    for x, y,l,index in data_loader:
        x= x.to(device)
        y= y.to(device)
        y_pred = model(x,l)
        pred = torch.max(y_pred, 1)[1]   
        correct += (pred.to(device) == y).sum()
        total += y.shape[0]
    return correct/total

def train_model(model, epochs=20, lr=0.001):
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        print(f'Epoch:{i+1}')
        for x, y,l,index in train_loader:
            x= x.to(device)
            y=y.to(device)
            y_pred = model(x,l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()

        train_acc=compute_accuracy(model,train_loader)
        print(f'Train_accuracy:{train_acc*100}')


model = RNN(vocab_size, 300, 128, embeddings).to(device)
train_model(model, epochs=25, lr=0.001)


print(f'Test accuracy:{compute_accuracy(model,test_loader)*100}')

# Test accuracy:0.4740350842475891 -with dropout
# Test accuracy:0.4373684227466583-without dropout

predictions={'Id':[],'Genre':[]}
predict=[]

for x,y,l,index in test_loader:
    x=x.to(device)
    y_pred= model(x,l)
    pred = torch.max(y_pred, 1)[1] 
    
    for i in range(len(index)):
        predictions['Id'].append(index[i].item())
        predictions['Genre'].append(pred[i].item())
   
predictions= pd.DataFrame(predictions)
predictions.to_csv(sys.argv[1]+'/non_comp_test_pred_y.csv',index=False)