import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients
from sklearn.metrics import accuracy_score, recall_score,f1_score
from sklearn.model_selection import train_test_split
from scipy.special import jn_zeros
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from ..auto import BaseDetector
from tqdm import tqdm
import os

from copy import deepcopy
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


class Toymodel(nn.Module):
    def __init__(self, in_dim):
        super(Toymodel, self).__init__()
        self.lin1 = nn.Linear(in_dim, 256)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(64, 2)

    def forward(self, x):
        return self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(x)))))
 

class IG_block:
    def __init__(self):
        super(IG_block, self).__init__()
        self.model = Toymodel(768).cuda()
        self.IG = IntegratedGradients(self.model)

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def inverse(self, embeds, tar, max_features=20):
        baseline = torch.zeros_like(embeds)
        ig_attrs, _ = self.IG.attribute(inputs=embeds, baselines=baseline, target=tar, n_steps=200,
                                        return_convergence_delta=True)
        max_ids = torch.argsort(torch.abs(ig_attrs), dim=1)[:, -max_features:].detach().cpu().numpy().tolist()
        B_size = embeds.shape[0]
        H_size = embeds.shape[1]
        embeds = embeds.expand(20, B_size, H_size)
        embeds = embeds.permute(1, 0, 2)
        #print(B_size)
        for i in range(B_size):
            for j in range(max_features):
                idx = max_ids[i][j]
                indices = (torch.LongTensor([i]), torch.LongTensor([j]), torch.LongTensor([idx]))
                embeds = deepcopy(embeds.index_put(indices, torch.Tensor([0.0]).cuda()))
        return embeds

class DEMASQ(nn.Module):
    def __init__(self):
        super(DEMASQ, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.IG = IG_block()

    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path,'demasq.pt'))
        torch.save(self.IG.model.state_dict(), os.path.join(path,'demasq_ig.pt'))
    
    def load(self, path):
        model_path = os.path.join(path,'demasq.pt')
        self.load_state_dict(torch.load(model_path))
        ig_path = os.path.join(path,'demasq_ig.pt')
        self.IG.model.load_state_dict(torch.load(ig_path))


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x



class DemasqDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        self.tokenizer = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        self.model = DEMASQ()
        self.model.cuda()
        self.sigmod = nn.Sigmoid()
        #TODO: state loading
        if "state_dict_path" in kargs:
            self.model.load(kargs['state_dict_path'])
        
    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        inputs = self.tokenizer.encode(text, convert_to_tensor=True)
        for emb in inputs:
            emb = emb.cuda()
            pred = self.sigmod(self.model(emb)).item()
            result.append(1-pred)
        return result if isinstance(text, list) else result[0]
    
    def finetune(self, data, f_config):
        X_train, X_val, y_train, y_val = train_test_split(data['text'], data['label'], test_size=0.1, random_state=42)
        train_embs = self.tokenizer.encode(X_train, convert_to_tensor=True)
        val_embs = self.tokenizer.encode(X_val, convert_to_tensor=True)
        train, val = list(zip(train_embs, y_train)), list(zip(val_embs, y_val))
        optimizer = optim.Adam(list(self.model.parameters())+list(self.model.IG.model.parameters()), lr=0.0001)
        criterion = nn.BCELoss()
        batch_size = f_config.batch_size
        num_epochs = f_config.epoch
        saved_path = f_config.save_path
        for epoch in range(num_epochs):
            running_loss = 0.0
            self.model.train()
            self.model.IG.train()
            for inputs, targets in tqdm(DataLoader(train, batch_size)):  # Loop over your data
                optimizer.zero_grad()  # Zero the gradients
                inputs = inputs.cuda()
                inputs_perm = self.model.IG.inverse(inputs, targets)
                e = enery(inputs_perm,targets) - min(enery(inputs_perm, 1), enery(inputs_perm, 0))
                e = e.cuda()
                targets = torch.Tensor([[1-targets]]).cuda()
                outputs = self.model(inputs)  # Forward pass
                loss = criterion(self.sigmod(outputs), targets) + e
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                running_loss += loss.item()
            epoch_loss = running_loss / len(train)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
            self.model.eval()
            self.model.IG.eval()
            preds = []
            labels = []
            for inputs, targets in DataLoader(val):  # Loop over your data
                inputs = inputs.cuda()
                labels.append(1-targets)
                targets = torch.Tensor([[1-targets]]).cuda()
                pred = self.sigmod(self.model(inputs))
                preds.append(1 if pred>0.5 else 0)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val: f1", f1_score(labels, preds))
        if f_config.need_save:
            self.model.save(saved_path)



  