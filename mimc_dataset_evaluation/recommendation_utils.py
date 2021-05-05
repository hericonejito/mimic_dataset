import torch
import numpy as np
import operator
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
def most_similar_items(drug_id,train_set): # train_set is a numpy array
    # we check the interest index in the unique list, which was used to build the matrix
    unique_drugs = train_set.shape[1]
    interest_sim = dict()
    
    for i in range(unique_drugs): #we take each interest and compute the similarity to current interest
        if i != drug_id:
            #print("Similarity between ", drug_id, " and ", i)
            #print(f"Drug {i} is {list(train_array[:,i])}")
            sim = drug_similarities[drug_id,i]
        else:
            sim = 0
        interest_sim[i] = np.round(sim,3)
    #print(interest_sim)
    return interest_sim



def get_ib_recommendations(user_id,train_set,neighborhood,top_n,similarities):
    user_drugs = np.where(train_set[user_id,:]==1)[0]
    drug_similarity_ranking = np.zeros(shape=(train_set.shape[1]))
    for drug in user_drugs:
        drug_similarity_ranking +=similarities[drug]
        
    #We need to remove drugs that are already taken by the current patient
    for drug in user_drugs:
        drug_similarity_ranking[drug] = -1
    return list(np.argsort(drug_similarity_ranking)[-top_n:])


def get_pop_recommendations(user_id, train_set, neighborhood,top_n,similarities):
    user_drugs = np.where(train_set[user_id,:]==1)[0]
    recommendations = train_set.sum(0)
    recommendations[user_drugs] = -10
    recommendations = np.argsort(recommendations)
    return recommendations[-top_n:]
#get_ib_recommendations(1,np.asarray(train_set),5)
def get_mf_recommendations(user_id,train_set,neighborhood,top_n,similarities,):
    user_drugs = np.where(train_set[user_id,:]==1)[0]
    current_user_similarities = similarities[user_id]
    current_user_similarities[user_drugs] = -1
    sorted_drugs_per_user = np.argsort(current_user_similarities)
    
        
    return sorted_drugs_per_user[-top_n:]


def return_rec(x):
        return x[0]
        
    
def get_most_similar_patient(user_id,similarities,neighborhood =5):
        similarities = np.asarray(similarities[user_id])
        return np.argsort(similarities)[-(neighborhood+1):-1]  
        

def get_ub_recommendations(user_id,train_set, user_neighborhood,top_n,similarities):
        similar_users = get_most_similar_patient(user_id,similarities,user_neighborhood)
        
        a = np.where(np.asarray(train_set)[similar_users]==1)
        unique, counts = np.unique(a[1], return_counts=True)
        recommendations = dict(zip(unique, counts))
        #We need to remove from recommendations drugs that are already in patient records 
        user_drugs = list(np.where(train_set[user_id]==1)[0])
        for drug in user_drugs:
            if drug in list(recommendations.keys()):
                recommendations.pop(drug)
        sorted_x = sorted(recommendations.items(), key=operator.itemgetter(1))
        results = map(return_rec,sorted_x)
        return list(results)[-top_n:]
def get_nn_recommendations(user_id,train_set,user_neighborhood,top_n,similarities,net):
    user_drugs = np.where(train_set[user_id,:]==1)[0]
    with torch.no_grad():
        user_recommendations = net((user_id*torch.ones((train_set.shape[1]))).long(),torch.Tensor(np.arange(0,train_set.shape[1])).long())
    user_recommendations = user_recommendations.cpu().numpy()
    #print(user_recommendations)
    user_recommendations[user_drugs]=-1
    
    sorted_recommendations = np.argsort(user_recommendations,axis=0)
    return sorted_recommendations[-top_n:]
def get_random_recommendations(user_id,train_set,top_n):
    user_drugs = np.where(train_set[user_id,:]==0)[0]
    random_choices =np.random.choice(user_drugs, size=top_n)
    return random_choices
def get_simrank_recommendations(user_id,train_set,top_n,similarities):
    user_drugs = np.where(train_set[user_id,:]==1)[0]
    drug_similarity_ranking = np.zeros(shape=(train_set.shape[1]))
    for drug in user_drugs:
        drug_similarity_ranking +=similarities[drug]
        drug_similarity_ranking[drug] = -10
    #We need to remove drugs that are already taken by the current patient
    
    return list(np.argsort(drug_similarity_ranking)[-top_n:])

    