import scipy.sparse as sp
from torch import sparse
import torch
import scipy
import numpy as np
from    torch.nn import functional as F
import random
import sys
import math
import pickle as pkl
from random import shuffle
import re

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
def shuffle_multi(a,b,c):
    total=list(zip(a,b,c))
    shuffle(total)
    a,b,c=zip(*total)
    return a,b,c

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = (mx.row, mx.col)
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = sparse_mx

    return sparse_mx
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def get_A_(adj:sp.csr_matrix):
    degree_matrix=adj.sum(1).T.tolist()[0]
    dialog=np.power(degree_matrix,-0.5).flatten()
    dialog[np.isinf(dialog)]=0.
    row=col=[i for i in range(len(dialog))]
    dialog=sp.csr_matrix((dialog,(row,col)))
    A_=sp.csr_matrix(dialog).dot(sp.csr_matrix(adj.T)).dot(sp.csr_matrix(dialog))
    return A_
def convert_sparse_to_tensor(matrix):
    # csr_matrix=matrix.to_coo()
    csr_matrix=matrix.tocoo()
    csr_matrix = torch.sparse.LongTensor(torch.LongTensor([csr_matrix.row.tolist(), csr_matrix.col.tolist()]),
                                         torch.FloatTensor(csr_matrix.data.astype(np.float)))
    return csr_matrix
def to_sparse(matrix):
    """

    :param matrix: dense torch tensor,shape(node_size,node_size)
    :return: sparse tensor
    """
    indices=torch.nonzero(matrix).t()#get not zero index
    value=matrix[tuple(indices[i] for i in range(indices.shape[0]))]
    sparse_matrix=sparse.FloatTensor(indices,value,matrix.size())
    return sparse_matrix
def split_train_test(vocab_num,doc_num,labels):
    train_num=int(doc_num*0.9)
    total_index=[i for i in range(vocab_num,vocab_num+doc_num)]
    train_index=total_index[:train_num]
    test_list=total_index[train_num:]
    train_label=labels[:train_num]
    test_label=labels[train_num:]

    return train_index,test_list,train_label,test_label
# a=torch.eye(12)
def convert_string_to_word(sentence:str):
    words=[]
    begin=True
    is_char=False
    digital=False
    sentence=sentence.strip()
    for char in sentence:
        if begin:
            words.append('')
            begin=False
        if char==' ':
            begin=True
            continue
        if char.isalnum():
            if digital:
                words[-1]+=char
            else:
                words.append('')
                words[-1]+=char
            digital=True
            is_char=False
        elif char.isalpha():
            digital=False
            if is_char:
                words[-1]+=char
            else:
                words.append('')
                words[-1]+=char
            is_char=True
        else:
            digital=False
            is_char=False
            # words.append(char)
            begin=True
    return words
def loss_function(predict,target):
    # predict=predict[index]
    target=target.unsqueeze(-1)
    predict_prob=torch.gather(predict,dim=1,index=target)
    predict_prob=torch.log(predict_prob)
    return -predict_prob.sum()
def loss_function2(predict,target,mask):
    # predict=predict[index]

    target=torch.argmax(target,-1)
    # target=target.unsqueeze(-1).long()
    # print("target.shape",target.shape)
    # print("predict.shape{}".format(predict.shape))
    # exit(0)

    # predict_prob=torch.gather(predict,dim=1,index=target)
    # predict_prob=torch.log(predict_prob)
    predict_prob=F.cross_entropy(predict,target,reduction='none')
    mask2=mask/torch.mean(mask.float())
    predict=predict_prob*mask2
    return predict.mean()
# print(to_sparse(a))

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    print(len(labels))

    train_idx_orig = parse_index_file(
        "../data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

