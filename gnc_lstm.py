import torch
print('Torch v: ', torch.__version__)

import numpy as np
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings('ignore')



from sktime.datasets import load_from_arff_to_dataframe
from sktime.datasets import load_from_tsfile
from sklearn.preprocessing import LabelEncoder

X_train, y_train = load_from_arff_to_dataframe("SelfRegulationSCP1_TRAIN.arff")
X_test, y_test = load_from_arff_to_dataframe("SelfRegulationSCP1_TEST.arff")

print(X_train.shape)
print(X_test.shape)
print(type(X_train.iloc[1, 1]))

print((X_train.iloc[1, 1].shape))
X_train.head()


#Convert pandas series into numpy array
X_train = np.array(X_train.values.tolist())
X_test = np.array(X_test.values.tolist())

#Convert alphabet label into numeric
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
print(y_train)

print("Train shape: ", X_train.shape, "  type: ", type(X_train))
print("Test shape: ", X_test.shape, "  type: ", type(X_test))
print("Train Label shape: ", y_train.shape, "  type: ", type(y_train))
print("Test Label shape: ", y_test.shape, "  type: ", type(y_test))

# The class labels
np.unique(y_train)

#check
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
y_train_stats = dict(zip(unique_y_train, counts_y_train))
print("y_train_counts")
print(y_train_stats)

unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
y_test_stats = dict(zip(unique_y_test, counts_y_test))
print("y_test_counts")
print(y_test_stats)


#graph utils
from sklearn.preprocessing import StandardScaler
def get_adj_mat(c, th):
  #print("Creating graph with th: ", th)
  n = c.shape[0]
  a = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      #print("before:", c[i,j])
      if(c[i,j]>th):
        a[i,j]=1
        a[j,i]=1
      #print("after:", a[i,j])
  return a

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def build_edge_index_tensor(adj):
  num_nodes = adj.shape[0]
  source_nodes_ids, target_nodes_ids = [], []
  for i in range(num_nodes):
    for j in range(num_nodes):
      if(adj[i,j]==1):
        source_nodes_ids.append(i)
        target_nodes_ids.append(j)
  edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))
  edge_index_tensor = torch.from_numpy(edge_index)
  return edge_index_tensor

def normalize_node_attributes(mvts):
  sc = StandardScaler()
  mvts_std = sc.fit_transform(mvts)
  return mvts_std

#data crawler in train dataset
th = 0
num_train = X_train.shape[0]
num_nodes = 6
num_ts = 896
train_adjs = np.zeros((num_train, num_nodes, num_nodes))
train_nats = np.zeros((num_train, num_nodes, num_ts))
for i in range(num_train):
  #print('Event: ', i)
  mt = X_train[i].T[:,:] #consider first 25 solar params
  mt = normalize_node_attributes(mt) #[60,25]
  c_mt = np.corrcoef(mt.T)#[25,25]
  c_mt[np.isnan(c_mt)]=0
  train_nats[i,:,:] = mt.T
  adj = get_adj_mat(c_mt, th)
  train_adjs[i,:,:]=adj

#data crawler in test dataset
num_test = X_test.shape[0]
#data crawler in train dataset
test_adjs = np.zeros((num_test, num_nodes, num_nodes))
test_nats = np.zeros((num_test, num_nodes, num_ts))
for i in range(num_test):
  #print('Event: ', i)
  mt = X_test[i].T[:,:] #consider first 25 solar params
  mt = normalize_node_attributes(mt) #[60,25]
  c_mt = np.corrcoef(mt.T)#[25,25]
  c_mt[np.isnan(c_mt)]=0
  test_nats[i,:,:] = mt.T #[25,60]
  adj = get_adj_mat(c_mt, th)
  test_adjs[i,:,:]=adj

print(train_adjs.shape)
print(train_nats.shape)
print(test_adjs.shape)
print(test_nats.shape)

#MODELS CELL
#node_emb_dim = graph_emb_dim = window_emb_dim = 4; sequence_emb_dim = 128; class_emb_dim = 4
# (GCN) Node emb -> (mean) Graph emb -> (Flatten, Linear) -> window emb -> (LSTM) -> Temporal sequence emb -> (Linear) Class emb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class MVTS_GCN_RNN(torch.nn.Module):
  def __init__(self, num_nodes, input_dims, device, sequence_emb_dims, gcn_hidden_dims, node_emb_dims, graph_emb_dims, event_emb_dims, num_classes):
    super(MVTS_GCN_RNN, self).__init__()
    self.num_nodes = num_nodes
    self.input_dims = input_dims
    self.device = device
    self.sequence_emb_dims = sequence_emb_dims
    self.gcn_hidden_dims = gcn_hidden_dims
    self.node_emb_dims = node_emb_dims
    self.graph_emb_dims = graph_emb_dims
    self.num_classes = num_classes

    self.mt2vector = nn.LSTM(num_nodes, sequence_emb_dims)
    self.conv1 = GCNConv(input_dims, gcn_hidden_dims)
    self.conv2 = GCNConv(gcn_hidden_dims, node_emb_dims)
    #self.conv = GCNConv(input_dims, node_emb_dims)
    #self.node2graph = nn.Linear(num_nodes*node_emb_dims, graph_emb_dims)#change from ex 1
    self.seqGraph2event = nn.Linear(sequence_emb_dims+graph_emb_dims, event_emb_dims)
    self.sequence2class_space = nn.Linear(event_emb_dims, num_classes)

  def forward(self, adj_mat, node_att):
    #node_att: [25, 60], adj_mat: [25, 25]
    #prepare for gcnconv
    edge_index_tensor = build_edge_index_tensor(adj_mat)
    edge_index = edge_index_tensor.to(self.device)
    node_attributes_tensor = torch.from_numpy(node_att)
    x = node_attributes_tensor.to(self.device)#[25,60]
    #lstm on x.T
    event_mvts = torch.t(x)#[60,25]
    event_vectors, _ = self.mt2vector(event_mvts.view(len(event_mvts), 1, -1))
    last_event_vector = event_vectors[len(event_vectors)-1]
    #GCN on graph
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index)
    x = F.relu(x)
    x = self.conv2(x, edge_index)
    #graph embedding
    x = torch.mean(x, dim=0).view(1,-1)
    graph_vector = x
    seq_graph_vector = torch.cat((last_event_vector, graph_vector), dim=1)#[128+16]
    event_vector = self.seqGraph2event(seq_graph_vector)#[1,128]
    event_vector = F.relu(event_vector)
    class_vector = self.sequence2class_space(event_vector)#[1,4]
    class_scores = F.log_softmax(class_vector, dim=1)
    return class_scores

#Training
torch.manual_seed(0)

NUM_NODES = 6
INPUT_DIMS = 896
GCN_HIDDEN_DIMS = 2 #kIPF used 4 hidden dims for karate (34, 154)
NODE_EMB_DIMS = 2 # number of classes/can be tuned
GRAPH_EMB_DIMS = NODE_EMB_DIMS #change from ex 1
EVENT_EMB_DIMS = 128 #number of sparsity threshold/can be increased #change from ex 1 #change from ex 10
SEQUENCE_EMB_DIMS = 128 #number of timestamps #change from ex 1 #change from ex 10
NUM_CLASSES = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = MVTS_GCN_RNN(NUM_NODES, INPUT_DIMS, device, SEQUENCE_EMB_DIMS, GCN_HIDDEN_DIMS, NODE_EMB_DIMS, GRAPH_EMB_DIMS, EVENT_EMB_DIMS, NUM_CLASSES).to(device).double()

loss_function = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01) #change from ex 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
num_epochs = 100 #change from ex 10

#Train
for epoch in range(num_epochs):
  #print('Epoch: ', epoch)
  for i in range(num_train):#num_train
    optimizer.zero_grad()
    #print('Event: ', i)
    adj_mat = train_adjs[i,:,:]#(25,25)
    node_att = train_nats[i,:,:] #(25,60)
    class_scores = model(adj_mat, node_att)
    target = [y_train[i]]
    target = torch.from_numpy(np.array(target))
    target = target.to(device)
    loss = loss_function(class_scores, target)
    loss.backward()
    optimizer.step()
  if(epoch%5==0):
    print ("epoch n loss:", epoch, loss)

#test accraucy
num_test = X_test.shape[0]
with torch.no_grad():
  numCorrect = 0
  for i in range(num_test):
    adj_mat = test_adjs[i,:,:]
    node_att = test_nats[i,:,:]
    test_class_scores = model(adj_mat, node_att)
    #test_mvts = X_test[i,:,:]
    test_label = y_test[i] #class = 2
    #test_class_scores = model(test_mvts) #test mvts = [0.35, 0.15, 0.45, 0.05]
    class_prediction = torch.argmax(test_class_scores, dim=-1) #2
    if(class_prediction == test_label): #(2,3 ) match
      numCorrect = numCorrect + 1
  acc = numCorrect/num_test
  print(acc)

#train acc
num_train = X_train.shape[0]
with torch.no_grad():
  numCorrect = 0
  for i in range(num_train):
    adj_mat = train_adjs[i,:,:]
    node_att = train_nats[i,:,:]
    train_class_scores = model(adj_mat, node_att)
    #test_mvts = X_test[i,:,:]
    train_label = y_train[i] #class = 2
    #test_class_scores = model(test_mvts) #test mvts = [0.35, 0.15, 0.45, 0.05]
    class_prediction = torch.argmax(train_class_scores, dim=-1) #2
    if(class_prediction == train_label): #(2,3 ) match
      numCorrect = numCorrect + 1
  print(numCorrect)
  acc = numCorrect/num_train
  print(acc)















