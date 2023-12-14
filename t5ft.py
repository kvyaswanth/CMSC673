import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = model.cuda()

import pandas as pd
import numpy as np

df = pd.read_csv('datasetSentences.csv')
labels = pd.read_csv('datasetSplit.csv')

def inp_ids(text):
  return tokenizer(text, return_tensors="pt", padding=True, truncation=True)['input_ids']

def token_types(text):
  return tokenizer(text, return_tensors="pt", padding=True, truncation=True)['token_type_ids']

def attn_mask(text):
  return tokenizer(text, return_tensors="pt", padding=True, truncation=True)['attention_mask']

df['input_ids'] = df['sentence'].apply(inp_ids)
df['attention_mask'] = df['sentence'].apply(attn_mask)

df['splitset_label'] = labels['splitset_label'].astype(str)
df['labels'] = df['splitset_label'].apply(inp_ids)

def extract_embeddings(row):
    inputs = {
        'input_ids': row['input_ids'].cuda(),
        'token_type_ids': row['token_type_ids'].cuda(),
        'attention_mask': row['attention_mask'].cuda()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings.cpu().numpy()
    
def pad_tensor(tensor, max_length):
    padding_length = max_length - tensor.shape[1]
    if padding_length > 0:
        pad =(0, 0, 0, padding_length)
        tensor = torch.nn.functional.pad(tensor, pad, "constant", 0)
    return tensor

def pad_attn_mask(tensor, max_length):
    padding_length = max_length - tensor.shape[1]
    if padding_length > 0:
        pad =(0, padding_length)
        tensor = torch.nn.functional.pad(tensor, pad, "constant", 0)
    return tensor


# Pad each tensor in the DataFrame
max_length = max([tensor.shape[1] for tensor in df.input_ids])
x1 = [pad_attn_mask(torch.tensor(tensor), max_length) for tensor in df.input_ids]
x1 = torch.vstack(x1)

max_length = max([tensor.shape[1] for tensor in df.attention_mask])
x2 = [pad_attn_mask(torch.tensor(tensor), max_length) for tensor in df.attention_mask]
x2 = torch.vstack(x2)


y = df['splitset_label'].values

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Instantiate SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to the data
X_resampled, y_resampled = smote.fit_resample(x1,y)
y_resampled = [inp_ids(label) for label in y_resampled]
y_resampled = torch.vstack(y_resampled)
xtrain, valtrain, ytrain, yval = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Instantiate SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(x2, y)
y_resampled = [inp_ids(label) for label in y_resampled]
y_resampled = torch.vstack(y_resampled)
masktrain, maskval, _, _ = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
batch_size = 1
learning_rate = 1e-5
num_epochs = 100



optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


train_inputs = torch.tensor(xtrain)
validation_inputs = torch.tensor(valtrain)

train_labels = torch.tensor(ytrain)
validation_labels = torch.tensor(yval)

train_masks = torch.tensor(masktrain)
validation_masks = torch.tensor(maskval)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#Creating the DataLoader which will help us to load data into the GPU/CPU
batch_size = 32*4

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)


total_loss = 0
# Training loop
for epoch in range(num_epochs):
    model.train()
    for idx,batch in enumerate(train_dataloader):
        inputs, attention_mask, targets = batch[0].cuda(),batch[1].cuda(),batch[2].cuda()
#         inputs, attention_mask, targets = batch#[0].cuda(),batch[1].cuda(),batch[2].cuda()
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=targets)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    print("epoch", epoch,  "| loss", loss)


preds = []
act = []
with torch.no_grad():
    for idx,batch in enumerate(train_dataloader):
        inputs, attention_mask = batch[0].cuda(),batch[1].cuda()
        optimizer.zero_grad()
        outputs = model.generate(inputs, attention_mask=attention_mask)
        for output in outputs:
            preds.append(tokenizer.decode(output, skip_special_tokens=True))
        for label in batch[2].numpy():
            act.append(tokenizer.decode(output, skip_special_tokens=True))

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(preds, act)
report = classification_report(preds, act)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

preds = []
act = []
with torch.no_grad():
    for idx,batch in enumerate(validation_dataloader):
        inputs, attention_mask = batch[0].cuda(),batch[1].cuda()
        optimizer.zero_grad()
        outputs = model.generate(inputs, attention_mask=attention_mask)
        for output in outputs:
            preds.append(tokenizer.decode(output, skip_special_tokens=True))
        for label in batch[2].numpy():
            act.append(tokenizer.decode(output, skip_special_tokens=True))

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(preds, act)
report = classification_report(preds, act)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
