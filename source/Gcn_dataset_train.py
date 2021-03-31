import torch
import time
from model2 import Model
from train2 import train
import scipy.sparse as sp
from utils import load_corpus,preprocess_adj
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
from train2 import test
import fitlog
from load_data import get_shuffle_data
np.set_printoptions(threshold= sys.maxsize)
setup_seed(3)
layer_num=2

device="cuda:0"
train_epoch=1000
path="../data"
epoch=1000
if path[-1] != '/':
    path = path + '/'
# vocab_name = path+'vocab.pkl'
# paper_index = path+'file.pkl'
# label_name = path+"label_dict.pkl"
matrix, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size=load_corpus('ohsumed')
# assert y_train ==y_val
# print("vocab size is{}".format(vocab_num))
# print("train size is{} test size is {}".format(len(train_index),len(test_index)))
# print("mask_total{}".format((val_mask & train_mask & test_mask).sum()))
A_=preprocess_adj(matrix)
train_mask=torch.tensor(train_mask).to(device)
test_mask=torch.tensor(test_mask).to(device)
# A_=torch.tensor(A_)
# print(type(A_))
# exit(0)
matrix=convert_sparse_to_tensor(A_)
# train_index,test_index,train_label,test_label=split_train_test(len(vocab.keys()),len(paper.keys()),labels)
# train_index=torch.tensor(y_train).to(device)
# test_index=torch.tensor(y_test).to(device)
print("y_train shape{}".format(y_train.shape))
print("y_test shape{}".format(y_test.shape))
train_label=torch.tensor(y_train).to(device)
test_label=torch.tensor(y_test).to(device)
val_index=[]
val_index=torch.tensor(val_index).to(device)
# val_label=torch.tensor(val_label).to(device)
feature=np.eye(matrix.shape[0],dtype=float)
feature=torch.tensor(feature).to(device)
layer_dims=[feature.shape[0],200,23]
model=Model(layer_num,layer_dims,feature.shape[0],23)
model=model.to(device)
optim=Adam(model.parameters(),0.02)
# train_label=torch.index_select(labels,0,train_index)
# test_label=torch.index_select(labels,0,test_index)
loss_not_decrease=0
best_loss=1e14
# loss_function=torch.nn.CrossEntropyLoss()
matrix = matrix.to(device).clone().detach()

best_accuracy=0
# orig_support=torch.sparse.mm(matrix.float(),feature.float())
for i in range(epoch):
    start_time=time.time()
    print(matrix.shape)
    accuracy,loss=train(matrix,feature,model,optim,i,device,train_mask,train_label,loss_function)
    if loss<best_loss:
        best_loss=loss
        loss_not_decrease=0
    else:
        loss_not_decrease+=1
    with torch.no_grad():
        # if len(val_index)!=0:
        #     eval_accuracy,eval_loss=test(matrix,feature,model,optim,i,device,val_index,val_label,loss_function,"eval")
        test_accuracy, test_loss = test(matrix, feature, model, optim, i, device,test_mask,
                                        test_label, loss_function,"test")
        if test_accuracy>best_accuracy:
            best_accuracy=test_accuracy
    print("epoch {} cost time {} s".format(i,time.time()-start_time))
    #
    # if loss_not_decrease>=10:
    #     break
print("best test accuracy is {}".format(best_accuracy))
