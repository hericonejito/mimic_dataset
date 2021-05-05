# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/20 0020 下午 8:33
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 按照GCN需要的格式 准备好邻接矩阵
"""
import numpy as np
import scipy.sparse as sp
import os
import sys
import pickle as pkl
from collections import Counter
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import dill
import gc
import pickle
import pickle as pkl
import pandas as pd

np.random.seed(1001)
train_split = 0.7
def init_ADJ(nodes_dict):
    adjacencies = []
    # n x n matrix (connected to each other node or not?)
    adj_shape = (len(nodes_dict), len(nodes_dict))

    edges = np.empty((len(nodes_dict), 2), dtype=np.int32)
    for j in range(len(nodes_dict)):
        edges[j] = np.array([j, j])
    row, col = np.transpose(edges)
    data = np.zeros(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)

    adjacencies.append(adj)
    adjacencies.append(adj)

    # Adjacency Matrix (initially all zeros)
    return adjacencies

def getADJ(action,seletectedActions,nodes_dict,ddiIDS):
    adjacencies=[]
    adj_shape=(len(nodes_dict),len(nodes_dict))

    spoList_rel0,spoList_rel1=get_spo(action, seletectedActions, ddiIDS)
    edges = np.empty((len(spoList_rel0), 2), dtype=np.int32)
    for j,(s,p,o) in enumerate(spoList_rel0):
        edges[j]=np.array([s,o])
    row,col=np.transpose(edges)
    data=np.ones(len(row))
    adj=sp.csr_matrix((data,(row,col)),shape=adj_shape,dtype=np.uint8)
    adjacencies.append(adj)

    # Processing the adjacency matrix of the counter tuples generated by the counter relation
    if len(spoList_rel1)==0:
        edges = np.empty((len(nodes_dict), 2), dtype=np.int32)
        for j in range(len(nodes_dict)):
            edges[j] = np.array([j, j])
        row, col = np.transpose(edges)
        data = np.zeros(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    else:
        edges = np.empty((len(spoList_rel1), 2), dtype=np.int32)
        for j, (s, p, o) in enumerate(spoList_rel1):
            edges[j] = np.array([s, o])
        row, col = np.transpose(edges)
        data = np.ones(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    adjacencies.append(adj)
    del adj,edges,adj_shape,data,row,col,spoList_rel0,spoList_rel1
    gc.collect()
    
    # Return the Medicine Knowledge Graph Representation  
    return adjacencies

# Subject-Predication-Object predictions
def get_spo(action,seletectedActions,ddiIDS):
    spoList0=[]
    # Add a combination drug group first
    spoList0.append([action,'组合',action])
    if len(seletectedActions)!=0:
        for id in seletectedActions:
            spoList0.append([action,'组合',id])
            spoList0.append([id,'组合',action])

    spoList1=[]
    for row in ddiIDS:
        if action==row[0]:
            # print('action-对抗-row[2]:',[action,'对抗',row[1]])
            spoList1.append([action,'对抗',row[1]])
        if action==row[1]:
            spoList1.append([row[0],'对抗',action])
    return spoList0,spoList1

def save_sparse_csr(filename,array):
    np.savez(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)

def load_data(filename):
    patients=[]
    drugs=[]
    drugSet=[]
    with open(filename,'r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            drugL=row[6].split(' ')
            row[4]=row[4].replace(' ','')
            tmpL=[item for item in drugL if item]
            if len(tmpL)>0:
                patients.append(' '.join(row[4]))
                drugs.append(tmpL)
                drugSet.extend(tmpL)
    return patients,drugs,list(set(drugSet))

def load_records(filename,cancer_data,dataset):
    if cancer_data:
        records = pd.read_csv('MIMIC-III/medical.txt', sep='\t', header=None)
        columns = ['user', 'treatment', 'drug', 'view', 'date', 'time']
        records.columns = columns


        x = records.groupby('user')['drug'].agg('unique')
        medicines = []
        for i in range(0, len(x)):
            if len(x.iloc[i]) > 1:
                user_medicine = f"{x.iloc[i][0]}"
                for drug in range(1,len(x.iloc[i])):
                    user_medicine += f' {x.iloc[i][drug]}'
                medicines.append(user_medicine)

        diagnosis = medicines
        procedures = medicines

    elif dataset == 'movielens':
        records = pd.read_csv('MIMIC-III/u.data',sep='\t',header = None)
        columns = ['user','drug','ranking','view']
        records.columns = columns
        medicines = []
        records = records[records['ranking'] > 0]
        records = records.drop('ranking', axis=1)

        x = records.groupby('user')['drug'].unique()
        for i in range(0, len(x)):
            if len(x.iloc[i]) > 20:
                user_medicine = f"{x.iloc[i][0]}"
                for drug in range(1,len(x.iloc[i])):
                    user_medicine += f' {x.iloc[i][drug]}'
                medicines.append(user_medicine)
        diagnosis = medicines
        procedures = medicines
    elif dataset == 'decagon':
        records = pd.read_csv('MIMIC-III/bio-decagon-targets.csv')
        columns = ['drug', 'user']
        records.columns = columns
        medicines = []
        x = records.groupby('user')['drug'].unique()
        for i in range(0, len(x)):
            if len(x.iloc[i]) > 15:
                user_medicine = f"{x.iloc[i][0]}"
                for drug in range(1, len(x.iloc[i])):
                    user_medicine += f' {x.iloc[i][drug]}'
                medicines.append(user_medicine)
        diagnosis = medicines
        procedures = medicines   
    else:
        records=dill.load(open(filename,'rb'))
        split_records = 20000
        diagnosis = [row[0] for row in records[:split_records]]
        procedures = [row[1] for row in records[:split_records]]
        medicines = [row[2] for row in records[:split_records]]
        medicines_temp =[]
        for medicine in medicines:
            if len(medicine.split(" "))>15:
                medicines_temp.append(medicine)
        medicines = medicines_temp
    # ID the code
    
    # Adding split records for taking few lines and not entire EHR dataset

    
    #diagnosis=[row[0] for row in records]
    #procedures=[row[1] for row in records]
    #medicines=[row[2] for row in records]
    
    # print ATC codes of drugs used on first 10 patients on record file
    print('medicines:',medicines[:10])
    # record with the most diagnosis and procedures values
    diagnosis_maxlen=max([len(line.split(' ')) for line in diagnosis])
    procedure_maxlen=max([len(line.split(' ')) for line in procedures])
    medicines_maxlen = max([len(line.split(' ')) for line in medicines])
    # Creating a bag-of-words for Diagnosis and Procedures sets
    # Each diagnosis/procedure code is taken from the EHR input file
    diagnosis_tokenizer = Tokenizer()
    diagnosis_tokenizer.fit_on_texts(diagnosis)
    sequences = diagnosis_tokenizer.texts_to_sequences(diagnosis)

    diagnosis_= pad_sequences(sequences, maxlen=diagnosis_maxlen, padding='post', truncating='post')

    procedure_tokenizer=Tokenizer()
    procedure_tokenizer.fit_on_texts(procedures)
    sequences=procedure_tokenizer.texts_to_sequences(procedures)

    procedure_=pad_sequences(sequences,maxlen=procedure_maxlen,padding='post',truncating='post')

    medicines_tokenizer = Tokenizer()
    medicines_tokenizer.fit_on_texts(medicines)
    sequences = medicines_tokenizer.texts_to_sequences(medicines)

    medicines_ = pad_sequences(sequences, maxlen=medicines_maxlen, padding='post', truncating='post')
    medicineSet=[]
    for row in medicines:
        for item in row.split(' '):
            if item not in medicineSet:
                medicineSet.append(item)

    drug2id={drug:id for id,drug in enumerate(medicineSet)}
    drug2id['END']=len(drug2id)
    drugIds=[]
    for line in medicines:
        line=line+' '+'END'
        drugIds.append([drug2id[item] for item in line.split(' ')])

    X,Y=[],[]
    for x,z in zip(medicines_,drugIds):
        train_number = int((len(z)-1)*0.7)
        x[train_number:] = 0
        X.append([x])
        Y.append(z[train_number:])
    print('drugIds:',drugIds[:5])
    return medicines_maxlen,procedure_maxlen,medicines_tokenizer,procedure_tokenizer,X,Y,drugIds,drug2id

def load_drugDDI(ddi_file,TOPK,med2id,cancer_data):
    from collections import defaultdict
    import pandas as pd

    if cancer_data:
        ddi = []
        ddi_df = pd.read_csv(ddi_file, sep=';')
        for i in range(0,len(ddi_df)):
            drug1 = ddi_df.iloc[i]['DrugId1']
            drug2 = ddi_df.iloc[i]['DrugId2']
            if (str(drug1) in list(med2id.keys())) and (str(drug2) in list(med2id.keys())):
                ddi.append([med2id[str(drug1)],med2id[str(drug2)]])
        return ddi
    else:
        atc3_atc4_dic = defaultdict(set)
        for item in med2id.keys():
            # Only considering ATC Codes up to 4 characters (this decrease the variety of codes)
            atc3_atc4_dic[item[:4]].add(item)

        cid2atc_dic=defaultdict(set)

        cid_atc='drug-atc.csv'

        # Mapping atc values for each CID drug to the obtained drug2id
        with open(cid_atc, 'r') as f:
            for line in f:
                line_ls = line[:-1].split(',')
                cid = line_ls[0]
                atcs = line_ls[1:]
                for atc in atcs:
                    # If atc[:4] value exists in drug2id, then assign CID 12-char value from
                    # 'drug_atc.csv' to drug2id 4-char drug value
                    if len(atc3_atc4_dic[atc[:4]]) != 0:
                        cid2atc_dic[cid].add(atc[:4])

        # ddi load
        ddi_df = pd.read_csv(ddi_file)
        # fliter severe side effects
        ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(
            columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
        ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
        # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
        filter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
        ddi_df = filter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates().reset_index(drop=True)

        # Map existing CID drug codes to ddi_df TOPK frequent drug interactions
        ddi_df['STITCH 1']=ddi_df['STITCH 1'].map(cid2atc_dic)
        ddi_df['STITCH 2']=ddi_df['STITCH 2'].map(cid2atc_dic)
        drugDDI=[]
        for drug1,drug2 in zip(ddi_df['STITCH 1'],ddi_df['STITCH 1']):
            for item in drug1:
                for item2 in drug2:
                    if item!=item2 and [med2id.get(item),med2id.get(item2)] not in drugDDI:
                        drugDDI.append([med2id.get(item),med2id.get(item2)])
        return drugDDI

def get_torch_sparse_matrix(A,dev):
    '''
    A : list of sparse adjacency matrices
    '''

    newA=[]
    for row in A:
        idx = torch.LongTensor([row.tocoo().row, row.tocoo().col])
        dat = torch.FloatTensor(row.tocoo().data)
        newA.append(torch.sparse.FloatTensor(idx, dat, torch.Size([row.shape[0], row.shape[1]])).to(dev))
    del idx,dat
    # gc.collect()
    return newA