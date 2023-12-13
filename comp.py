from transformers import AutoTokenizer
import pandas as pd
import os
import torch
from datasets import Dataset, load_metric
import torch
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import TextClassificationPipeline

X_train = pd.read_csv(sys.argv[1] + "/train_x.csv")[['Title']]
Y_train = pd.read_csv(sys.argv[1] + "/train_y.csv")[['Genre']]

m = X_train.shape[0]

X_train = pd.concat([X_train, Y_train], axis = 1)

X_test = pd.read_csv(sys.argv[1] + "/non_comp_test_x.csv")[['Title']]
Y_test = pd.read_csv(sys.argv[1] + "/non_comp_test_y.csv")[['Genre']]

m_test = X_test.shape[0]

X_test = pd.concat([X_test, Y_test], axis = 1)

test = pd.read_csv(sys.argv[1] + "/comp_test_x.csv")

X_train = X_train.rename(columns = {'Title' : 'text', 'Genre' : 'label'})
X_test = X_test.rename(columns = {'Title' : 'text', 'Genre' : 'label'})

traindataset = Dataset.from_pandas(X_train)
testdataset = Dataset.from_pandas(X_test)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_traindata = traindataset.map(preprocess_function, batched=True)
tokenized_testdata = testdataset.map(preprocess_function, batched = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = load_metric('glue', 'mnli')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 30)

trainer = Trainer(
    model,
    args = training_args,
    train_dataset=tokenized_traindata,
    eval_dataset=tokenized_testdata,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device = 0)

def find_prediction(y):
    scores=[]
    for i in range(30):
      scores.append(y[i]['score'])
    return scores.index(max(scores))

predictions = {'Id':[],'Genre':[]}
for index, row in test.iterrows():
    pred = pipe(row['Title'])
    pred = find_prediction(pred[0])
    predictions['Id'].append(index)
    predictions['Genre'].append(pred)
  
df = pd.DataFrame(predictions)
df.to_csv(sys.argv[1] + "/comp_test_y.csv", index = False)
