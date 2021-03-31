import torch
import time
from model import Model
from train import train
import scipy.sparse as sp
from utils import loss_function2 as loss_function
from process_data import process_data
import os
import nltk
# nltk.download('stopwords')
import sys
from torch.optim import Adam
from utils import split_train_test,setup_seed,get_A_,convert_sparse_to_tensor
import pickle
import numpy as np
from train import test
import fitlog
from load_data import get_shuffle_data
np.set_printoptions(threshold= sys.maxsize)
setup_seed(1)
layer_num=2

device="cuda:1"
train_epoch=200
path="../data"
epoch=200
if path[-1] != '/':
    path = path + '/'
vocab_name = path+'vocab.pkl'
paper_index = path+'file.pkl'
label_name = path+"label_dict.pkl"
matrix_name="matrix.pkl"
label_file = path+"labels.pkl"
train_file=path+"train_index.pkl"
test_file=path+"test_index.pkl"
val_file=path+"val_index.pkl"
files=[path+"adj_matrix.pkl",path+"labels.pkl",
path+"train_index.pkl",
path+"test_index.pkl"]

# for file in files:
#     if not os.path.exists(file):
#         process_data(path)
#         break
# # if not (os.path.exists(label_file) and os.path.exists(label_name) and os.path.exists(
#         matrix_name) and os.path.exists(vocab_name) and os.path.exists(paper_index)and os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file)) :
#
#     process_data(path)

print("load_data")
# matrix,train_index,test_index,labels,vocab=get_shuffle_data()
matrix=pickle.load(open(matrix_name,'rb'))
train_index=pickle.load(open(train_file,'rb'))
test_index=pickle.load(open(test_file,'rb'))
labels=pickle.load(open(label_file,'rb'))
print(matrix.shape)
# vocab_num=le
val_index=[]
# val_index=pickle.load(open(val_file,'rb'))
# matrix = pickle.load(open(matrix_name, 'rb'))
# print("matrix shape {}".format(matrix.shape))
# vocab = pickle.load(open(vocab_name, 'rb'))
# paper = pickle.load(open(paper_index, 'rb'))
# label_dict = pickle.load(open( label_name,'rb'))
vocab_num=14157
# labels=pickle.load(open(label_file,'rb'))
val_label=[labels[i] for i in val_index]
# train_index=pickle.load(open(train_file,'rb'))
# test_index=pickle.load(open(test_file,'rb'))
train_label=[labels[i] for i in train_index]
# print(matrix.todense())
# exit(0)
test_label=[labels[i] for i in test_index]
# vocab_num=len(vocab.keys())
# train_index=[i+vocab_num for i in train_index]
# test_index=[i+vocab_num for i in test_index]
# val_index=[i+vocab_num for i in []]
print("vocab size is{}".format(vocab_num))
print("train size is{} test size is {}".format(len(train_index),len(test_index)))
A_=get_A_(matrix+sp.eye(matrix.shape[0]))
# A_=torch.tensor(A_)
matrix=convert_sparse_to_tensor(A_)
# train_index,test_index,train_label,test_label=split_train_test(len(vocab.keys()),len(paper.keys()),labels)
train_index=torch.tensor(train_index).to(device)
test_index=torch.tensor(test_index).to(device)
train_label=torch.tensor(train_label).to(device)
test_label=torch.tensor(test_label).to(device)
val_index=[]
val_index=torch.tensor(val_index).to(device)
val_label=torch.tensor(val_label).to(device)
feature=np.eye(vocab_num+len(train_index)+len(test_index)+len(val_index),dtype=float)
feature=torch.tensor(feature).to(device)
layer_dims=[feature.shape[0],200,23]
model=Model(layer_num,layer_dims,vocab_num+len(train_index)+len(test_index),23)
model=model.to(device)
optim=Adam(model.parameters(),0.02)
# train_label=torch.index_select(labels,0,train_index)
# test_label=torch.index_select(labels,0,test_index)
loss_not_decrease=0
best_loss=1e14
loss_function=torch.nn.CrossEntropyLoss()
matrix = matrix.to(device)

best_accuracy=0
# orig_support=torch.sparse.mm(matrix.float(),feature.float())
for i in range(epoch):
    start_time=time.time()
    print(matrix.shape)
    accuracy,loss=train(matrix,feature,model,optim,i,device,train_index,train_label,loss_function)
    if loss<best_loss:
        best_loss=loss
        loss_not_decrease=0
    else:
        loss_not_decrease+=1
    with torch.no_grad():
        if len(val_index)!=0:
            eval_accuracy,eval_loss=test(matrix,feature,model,optim,i,device,val_index,val_label,loss_function,"eval")
        test_accuracy, test_loss = test(matrix, feature, model, optim, i, device, test_index,
                                        test_label, loss_function,"test")
        if test_accuracy>best_accuracy:
            best_accuracy=test_accuracy
    print("epoch {} cost time {} s".format(i,time.time()-start_time))
    #
    if loss_not_decrease>=10:
        break
print("best test accuracy is {}".format(best_accuracy))
