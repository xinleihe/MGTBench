import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, recall_score
from scipy.special import jn_zeros

def compute_zeros(d, c_n):
    m = d
    n = c_n
    drumhead_fundamental_f = jn_zeros(0,1)
    drumhead_f = jn_zeros(m, n+1)
    return drumhead_f[n]//drumhead_fundamental_f

def source_frequency(emb):
    min_emb = torch.min(emb)
    max_emb = torch.max(emb)
    if min_emb<0:
        emb = emb + min_emb
    else:
        emb = emb - max_emb
    unique_values = torch.unique(emb)
    return compute_zeros(0, len(unique_values))

def enery(emb, y):
    c = torch.var(emb).cpu()
    v_r = (1-y)*abs(c)
    v_s = 0.8
    E_f0 = source_frequency(emb)
    E_f = (c+v_r)/(c-v_s)*E_f0
    return E_f


class DEMASQ(nn.Module):
    def __init__(self):
        super(DEMASQ, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


from mgtbench.loading.dataloader import load
from  sentence_transformers import SentenceTransformer
data = load('Essay', 'ChatGLM')
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
train_embs = model.encode(data['train']['text'], convert_to_tensor=True)
test_embs = model.encode(data['test']['text'], convert_to_tensor=True)
train = list(zip(train_embs, data['train']['label']))
test = list(zip(test_embs, data['test']['label']))

num_epochs = 12
from torch.utils.data import DataLoader
# Instantiate the model
demasq = DEMASQ()
demasq.cuda()
optimizer = optim.Adam(demasq.parameters(), lr=0.0001)
criterion = nn.BCELoss()
m = nn.Sigmoid()

from tqdm import tqdm
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for inputs, targets in tqdm(DataLoader(train)):  # Loop over your data
        optimizer.zero_grad()  # Zero the gradients
        e = enery(inputs,targets) - min(enery(inputs, 1), enery(inputs, 0))
        inputs = inputs.cuda()
        e = e.cuda()
        targets = torch.Tensor([[1-targets]]).cuda()
        
        outputs = demasq(inputs)  # Forward pass
        loss = criterion(m(outputs), targets) 
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()
    epoch_loss = running_loss / len(train)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
    model.eval()
    preds = []
    labels = []
    for inputs, targets in tqdm(DataLoader(test)):  # Loop over your data
        inputs = inputs.cuda()
        labels.append(1-targets)
        targets = torch.Tensor([[1-targets]]).cuda()
        pred = m(demasq(inputs))
        preds.append(1 if pred>0.5 else 0)
    print(accuracy_score(labels,preds), recall_score(labels,preds))
    
        