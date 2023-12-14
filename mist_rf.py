from transformers import AutoTokenizer, AutoModel
import transformers
import torch

#modelname = "google/flan-t5-xxl"
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
llama_model = model.cuda()
tokenizer.pad_token = tokenizer.eos_token

import pandas as pd
import numpy as np

df = pd.read_csv('datasetSentences.csv')
labels = pd.read_csv('datasetSplit.csv')
#df = df[:10]i
def inp_ids(text):
  return tokenizer(text, return_tensors="pt", padding=True, truncation=True)['input_ids']

def token_types(text):
  return tokenizer(text, return_tensors="pt", padding=True, truncation=True)['token_type_ids']

def attn_mask(text):
  return tokenizer(text, return_tensors="pt", padding=True, truncation=True)['attention_mask']


df['input_ids'] = df['sentence'].apply(inp_ids)
# df['token_type_ids'] = df['sentence'].apply(token_types)
df['attention_mask'] = df['sentence'].apply(attn_mask)


def extract_embeddings(row):
    inputs = {
        'input_ids': row['input_ids'].cuda(),
#         'token_type_ids': row['token_type_ids'].cuda(),
        'attention_mask': row['attention_mask'].cuda(),
	'output_hidden_states' : True
    }
    
    #inputs = {
     #   'input_ids': row['input_ids'],
#         'token_type_ids': row['token_type_ids'].cuda(),
      #  'attention_mask': row['attention_mask']
   # }
    

    with torch.no_grad():
        outputs = llama_model(**inputs)
        embeddings = outputs.hidden_states
        #print(len(embeddings))
        return embeddings[-1].cpu().numpy()


df['embeddings'] = df[['input_ids','attention_mask']].apply(extract_embeddings, axis=1)
df.to_csv('llama_embeddings.csv')
#df['llama_embeddings']
# Find the maximum sequence length
max_length = max([tensor.shape[1] for tensor in df.embeddings])

def pad_tensor(tensor, max_length):
    padding_length = max_length - tensor.shape[1]
    if padding_length > 0:
        pad =(0, 0, 0, padding_length)
        tensor = torch.nn.functional.pad(tensor, pad, "constant", 0)
    return tensor
# Pad each tensor in the DataFrame
X = [pad_tensor(torch.tensor(tensor), max_length) for tensor in df.embeddings]

# Now you can stack them
X = torch.vstack(X)
X = [tensor.mean(dim=0) for tensor in X]
X = torch.vstack(X)
torch.save(X, 'tensor.pt')

y = labels['splitset_label'].values
from imblearn.over_sampling import SMOTE

# Instantiate SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to the data
X_resampled, y_resampled = smote.fit_resample(X, y)


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier with 100 trees (you can adjust this parameter)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

df.to_csv('llama_in.csv')
