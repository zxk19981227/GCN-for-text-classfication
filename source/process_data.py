import os
from collections import defaultdict
import math
import pickle
from  tqdm import tqdm
from random import shuffle
from utils import convert_string_to_word,clean_str
import scipy.sparse as sp
from nltk.corpus import stopwords
from utils import shuffle_multi

def get_data(path):
    train_data_set=os.listdir(path+'/training')
    # print(train_data_set)
    train_index=[]
    test_index=[]
    labels_dict={}
    for i,each in enumerate(train_data_set):
        labels_dict[each]=i
    train_data=[]
    train_label=[]
    file_names=defaultdict(int)
    train_file_names=[]
    test_file_names=[]
    for name in train_data_set:
        current_path=path+'/training/'+name
        current_label=labels_dict[name]
        files=os.listdir(current_path)
        for each in files:
            file_name=current_path+'/'+each
            file_data=open(file_name).readlines()
            file_data=" ".join(file_data)
            train_data.append(file_data)
            train_label.append(current_label)
            train_file_names.append(each)
            # file_names[each]=len(file_names.keys())
            # train_index.append(file_names[each])
    train_file_names,train_label,train_data=shuffle_multi(train_file_names,train_label,train_data)

    test_data=[]
    test_label=[]
    for name in train_data_set:
        current_path=path+'/test/'+name
        current_label=labels_dict[name]
        files=os.listdir(current_path)
        for each in files:
            file_name=current_path+'/'+each
            file_data=open(file_name).readlines()
            file_data=' '.join(file_data)
            test_data.append(file_data)
            test_label.append(current_label)
            # file_names[each]=len(file_names.keys())
            test_file_names.append(each)
            # test_index.append(file_names[each])
    test_file_names,test_label,test_data=shuffle_multi(test_file_names,test_label,test_data)
    data=[]
    label=[]
    for i,(a,b,c) in enumerate(zip(train_file_names,train_label,train_data)):
        file_names[a]=len(file_names.keys())
        label.append(b)
        data.append(c)
        train_index.append(i)
    train_num=len(train_index)
    train_num=int(train_num)
    val_index=train_index[train_num:]
    train_index=train_index[:train_num]
    for i,(a,b,c) in enumerate(zip(test_file_names,test_label,test_data)):
        file_names[a]=len(file_names.keys())
        label.append(b)
        data.append(c)
        test_index.append(i+train_num)
    return data,label,file_names,labels_dict,train_index,test_index,val_index
def convert_data_to_index(data):
    word_index={}
    index_data=[]
    delete_list=set()
    stop_words=set(stopwords.words('english'))
    word_fluence=defaultdict(int)
    for line in data:
        # print(line)
        line=clean_str(line)
        line=line.split(' ')
        for word in line:
            word=word.lower()
            if word in stop_words:
                delete_list.add(word)
            else:
                word_fluence[word]+=1
    for word in word_fluence.keys():
        if word_fluence[word]<5:
            delete_list.add(word)
    for line in data:
        current_data=[]
        line=clean_str(line)
        line=line.split(' ')
        for word in line:
            word=word.lower()
            if word in delete_list:
                continue
            if word not in word_index.keys():
                word_index[word]=len(word_index.keys())
            current_data.append(word_index[word])
        index_data.append(current_data)
    return index_data,word_index
def cal_matrix(path,file_names,data,word_index,window_size):
    paper_start=len(word_index.keys())
    paper2index={}
    index2paper={}
    col=[]
    row=[]
    weight=[]
    for each in file_names.keys():
        paper2index[each]=file_names[each]+paper_start
        index2paper[file_names[each]+paper_start]=each
    print("process docs finished")
    windows=[]
    for words in data:
        if len(words)<=window_size:
            windows.append(words)
        else:
            for i in range(len(words)-window_size+1):
                windows.append(words[i:i+window_size])
    print("converted doc to window finished")
    single_word_freq=defaultdict(int)
    for window in windows:
        appear=set()
        for word in window:
            if word not in appear:
                appear.add(word)
                single_word_freq[word]+=1
    tuple_dict=defaultdict(int)
    for window in tqdm(windows):
        for i in range(1,len(window)):
            for j in range(0,i):
                if window[i]==window[j]:
                    continue
                tuple_dict[(window[i],window[j])]+=1
                tuple_dict[(window[j],window[i])]+=1
    #cal TF_DIF
    print("converted tuple text finished")
    word_doc_freq_dict=defaultdict(int)
    word_doc_num=defaultdict(int)
    for file_name,each in zip(file_names,data):
        current_appear=set()
        for word in each:
            # print(paper2index[file_name,word])
            # print(word_index[paper2index])
            word_doc_freq_dict[word]+=1
            if word not in current_appear:
                current_appear.add(word)
                word_doc_num[word]+=1
    total_window_num=len(windows)
    print("converted doc to window finished")
    #计算pmi并且填充到矩阵中
    for each in tuple_dict.keys():
        i,j=each[0],each[1]
        frequence=tuple_dict[(i,j)]
        i_freq=single_word_freq[i]
        j_freq=single_word_freq[j]
        pmi=math.log(frequence*total_window_num/(i_freq*j_freq))
        if pmi<=0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)
    total_file_num=len(data)
    for file_name,doc in zip(file_names.keys(),data):
        current_word_num=len(doc)
        current_word_dict=defaultdict(int)
        for word in doc:
            current_word_dict[word]+=1
        current_file_num=paper2index[file_name]
        for word in doc:
            TF_IDF=current_word_dict[word]/current_word_num*math.log(total_file_num/(word_doc_num[word]+1))
            row.append(current_file_num)
            col.append(word)
            weight.append(TF_IDF)
    print("matrix finished")
    number_nodes=len(data)+len(word_index.keys())
    for i in range(number_nodes):
        col.append(i)
        row.append(i)
        weight.append(1)
    adj_matrix=sp.csr_matrix((weight,(row,col)),shape=(number_nodes,number_nodes))
    #这一步主要是生成对称矩阵
    # adj_matrix=adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix) - adj_matrix.multiply(adj_matrix.T > adj_matrix)
    pickle.dump(adj_matrix,open(path+"adj_matrix.pkl",'wb'))
    pickle.dump(word_index,open(path+"vocab.pkl",'wb'))
    pickle.dump(paper2index,open(path+'file.pkl','wb'))
    return adj_matrix
def process_data(path):
    data,label,file_names,labels_dict,train_index,test_index,val_index=get_data(path)
    # shuffle_data=list(zip(data,label,file_names))
    # shuffle(shuffle_data)
    # data,label,file_names=zip(*shuffle_data)
    pickle.dump(label,open(path+"labels.pkl",'wb'))
    pickle.dump(labels_dict,open(path+"label_dict.pkl",'wb'))
    pickle.dump(train_index,open(path+"train_index.pkl",'wb'))
    pickle.dump(test_index,open(path+"test_index.pkl",'wb'))
    pickle.dump(val_index,open(path+"val_index.pkl",'wb'))
    data,word_index=convert_data_to_index(data)
    matrix=cal_matrix(path,file_names,data,word_index,20)
    return matrix
