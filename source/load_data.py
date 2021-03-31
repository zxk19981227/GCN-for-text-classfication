from utils import clean_str
from  tqdm import tqdm
import pickle as pkl
import math
import scipy.sparse as sp
from nltk.corpus import stopwords
def convert_str_to_idx(data,doc_num):
    word_index={}
    stop_words=set(stopwords.words('english'))
    word_freq={}
    index_sen=[]
    for line in data:
        # print(line)
        line=line.split(' ')
        for word in line:
            if word in word_freq.keys():
                word_freq[word]+=1
            else:
                word_freq[word]=1
    for line in data:
        current_feq=[]
        line=line.split(' ')
        for word in line:
            if word_freq[word]>=5 and word not in stop_words:
                if word not in word_index.keys():
                    word_index[word]=len(word_index.keys())+doc_num
                current_feq.append(word_index[word])
        index_sen.append(current_feq)
    return index_sen,word_index

def load_data():
    label_idct={}
    with open("./ohsumed_shuffle.txt") as f:
        lines=f.readlines()
        train_index=[]
        test_index=[]
        # val_index=[]
        file=[]
        labels=[]
        for i in range(len(lines)):
            line=lines[i].strip().split('\t')
            file_name,usage_set,label=line
            if label not in label_idct.keys():
                label_idct[label]=len(label_idct.keys())
            file.append(file_name)
            labels.append(label_idct[label])
            if usage_set=='training':
                train_index.append(i)
            elif usage_set=='test':
                test_index.append(i)
            else:
                print(usage_set)
                print("type error")
                exit(0)
        data=[]
        for file_name in file:
            lines=open(file_name).readlines()
            lines=[clean_str(i.strip()) for i in lines]
            # print(lines)
            # print(' '.join(lines))
            data.append(' '.join(lines))
        data,word_vocab=convert_str_to_idx(data,len(data))
    return data,labels,word_vocab,train_index,test_index

def create_matrix(data,vocab_num,window_size=20):
    windows=[]
    for line in data:
        if len(line)<=window_size:
            windows.append(line)
        else:
            for i in range(len(line)-window_size+1):
                windows.append(line[i:i+window_size])
    windows_num=len(windows)
    #这个函数是用来获得单个单词出现窗口的数目
    single_word_appear_window={}
    for window in tqdm(windows):
        appear=set()
        for word in window:
            if word not in appear:
                if word not in single_word_appear_window.keys():
                    single_word_appear_window[word]=0
                single_word_appear_window[word]+=1
                appear.add(word)
    #接下来开始计算两个单词的称对出现次数。参考原文代码采用str作为测试样本
    tuple_appear={}
    for window in tqdm(windows):
        for i in range(1,len(window)):
            for j in range(0,i):
                if window[i]==window[j]:
                    continue
                current_str=str(window[i])+','+str(window[j])
                if current_str not in tuple_appear.keys():
                    tuple_appear[current_str]=0
                tuple_appear[current_str]+=1
                current_str=str(window[j])+','+str(window[i])
                if current_str not in tuple_appear.keys():
                    tuple_appear[current_str]=0
                tuple_appear[current_str]+=1
    #这个函数是用来计算单词出现地文档数目
    word_doc_num={}
    for line in tqdm(data):
        appear=set()
        for word in line:
            if word not in appear:
                if word not in word_doc_num.keys():
                    word_doc_num[word]=0
                word_doc_num[word]+=1
                appear.add(word)
    #用来计算PMI
    row=[]
    col=[]
    score=[]
    for s in tuple_appear.keys():
        i,j=s.split(',')
        i,j=int(i),int(j)
        ij_total=tuple_appear[s]
        i_score=single_word_appear_window[i]
        j_score=single_word_appear_window[j]
        row.append(i)
        col.append(j)
        score.append(ij_total*windows_num/i_score/j_score)
    #计算TF-IDF
    doc_num=len(data)
    for i,doc in tqdm(enumerate(data)):
        word_num={}
        for word in doc:
            if word not in word_num.keys():
                word_num[word]=0
            word_num[word]+=1
        total_word=len(doc)
        for word in word_num.keys():
            row.append(i)
            col.append(word)
            score.append(math.log(doc_num/(word_doc_num[word]+1))*word_num[word]/total_word)
    for i in range(len(data)+vocab_num):
        row.append(i)
        col.append(i)
        score.append(1)
    sparse_matrix=sp.csr_matrix((score,(row,col)),shape=(doc_num+vocab_num,doc_num+vocab_num))
    return sparse_matrix
def get_shuffle_data():
    data, label, word_vocab, train_index, test_index=load_data()
    # data,vocab=convert_str_to_idx(data,len(data))
    matrix=create_matrix(data,len(word_vocab.keys()),20)
    pkl.dump(matrix,open("matrix.pkl",'wb'))
    pkl.dump(label,open("label.pkl",'wb'))
    pkl.dump(train_index,open("train.pkl",'wb'))
    pkl.dump(test_index,open("test.pkl",'wb'))
    return matrix,train_index,test_index,label,word_vocab