import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os
import multiprocessing

def get_target(triples,file_paths):
    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    ent2id_dict, ids = read_dict([file_paths + "/ent_ids_" + str(i) for i in range(1,3)])
    
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return r_hs, r_ts, ids

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    time = set([0])
    for line in open(file_name,'r'):
        params = line.split()
        if len(params) == 5:
         ##   head, r, tail, ts, te = int(params[0]), int(params[1]), int(params[2]), 0,0
            head, r, tail, ts, te = int(params[0]), int(params[1]), int(params[2]), int(params[3]), int(params[4])###      
            entity.add(head); entity.add(tail); rel.add(r+1); time.add(ts+1); time.add(te+1)
            triples.append([head,r+1,tail,ts+1,te+1])
        else:
         ##   head, r, tail, t = int(params[0]), int(params[1]), int(params[2]), 0### by setting all timestamps to 0, we get TU-GNN
            head, r, tail, t = int(params[0]), int(params[1]), int(params[2]), int(params[3])###
            entity.add(head); entity.add(tail); rel.add(r+1); time.add(t+1)###
            triples.append([head,r+1,tail,t+1])####
    return entity,rel,triples,time

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def get_matrix(triples,entity,rel,time): ###
        ent_size = max(entity)+1
        rel_size = (max(rel) + 1)
        time_size = (max(time)+1)   ###
        print(ent_size,rel_size,time_size)
        adj_matrix = sp.lil_matrix((ent_size,ent_size))
        adj_features = sp.lil_matrix((ent_size,ent_size))
        radj = []
        rel_in = np.zeros((ent_size,rel_size))
        rel_out = np.zeros((ent_size,rel_size))


        ########3
     
        for i in range(max(entity)+1):
            adj_features[i,i] = 1

        if len(triples[0])<5:
            time_link = np.zeros((ent_size,time_size))
            for h,r,t,tau in triples:      ###  
                adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
                adj_features[h,t] = 1; adj_features[t,h] = 1;
                radj.append([h,t,r,tau]); radj.append([t,h,r+rel_size,tau]);######
                time_link[h][tau] +=1 ; time_link[t][tau] +=1 ####
                rel_out[h][r] += 1; rel_in[t][r] += 1
        else:
            time_link = np.zeros((ent_size,time_size))
            for h,r,t,ts,te in triples:      ###  
                adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
                adj_features[h,t] = 1; adj_features[t,h] = 1;
                radj.append([h,t,r,ts]); radj.append([t,h,r+rel_size,te]);######
                time_link[h][te] +=1 ; time_link[h][te] +=1 ####
                time_link[t][ts] +=1 ; time_link[t][te] +=1 ####
                rel_out[h][r] += 1; rel_in[t][r] += 1
        count = -1
        s = set()
        d = {}
        r_index,t_index,r_val = [],[],[]
        
        for h,t,r,tau in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h),str(t)]) in s:
                r_index.append([count,r])
                t_index.append([count,tau])########
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h),str(t)]))
                r_index.append([count,r])
                t_index.append([count,tau])##########
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]
            

        time_features  = time_link
        time_features = normalize_adj(sp.lil_matrix(time_features))         
##########################################################################        
        rel_features = np.concatenate([rel_in,rel_out],axis=1)
        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(sp.lil_matrix(rel_features))    
        return adj_matrix,r_index,r_val,t_index,adj_features,rel_features,time_features###
    
def load_data(lang,train_ratio = 1000):             
    entity1,rel1,triples1,time1 = load_triples(lang + 'triples_1')
    entity2,rel2,triples2,time2 = load_triples(lang + 'triples_2')
    
    
    train_pair = load_alignment_pair(lang + 'sup_pairs')
    dev_pair = load_alignment_pair(lang + 'ref_pairs')
    dev_pair = train_pair[train_ratio:]+dev_pair
    train_pair = train_pair[:train_ratio]
    
    adj_matrix,r_index,r_val,t_index,adj_features,rel_features,time_features = get_matrix(triples1+triples2,
    entity1.union(entity2),rel1.union(rel2),time1.union(time2))######
    
    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),np.array(t_index),adj_features,rel_features,time_features###

def get_hits(vec, test_pair, wrank = None, top_k=(1, 5, 10)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)
    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i,sim[i,j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank,-1),np.expand_dims(wrank,-1)],-1),axis=-1)
        sim = rank.argsort(-1)
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:,i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_lr / Lvec.shape[0]))  
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_rl / Rvec.shape[0]))
