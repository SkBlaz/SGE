
"""
MetaProp: scalable symbolic learning from heterogeneous information networks via propositionalization
Skrlj, 2019
"""
from numba import jit,prange
import operator
import operator
import numpy as np
import networkx as nx
import random
#from sklearn.preprocessing import OneHotEncoder
from collections import Counter,defaultdict
import multiprocessing as mp
from scipy.stats import moyal,levy
import sys
from scipy import sparse
import scipy.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, HashingVectorizer
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import itertools
import queue
import logging

try:
    import pyfpgrowth
    fp_import = True
except:
    fp_import = False

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

class GarVectorizer:
    def __init__(self, min_support = 3,num_features=1000):
        self.min_support = min_support
        self.max_features = num_features

    def fit(self,transactions):
        logging.info("Starting pattern mining!")
        self.pattern_hash = sorted(dict(pyfpgrowth.find_frequent_patterns(transactions, self.min_support)).items(),key=operator.itemgetter(1),reverse=True)
        self.features = [k for k,v in self.pattern_hash][0:self.max_features]
        logging.info("Found rules: {}".format(len(self.features)))

    def fit_transform(self,transactions):
        self.fit(transactions)
        output_features = []
        for transaction in transactions:
            feature_vector = []
            for feature in self.features:
                cont = 0
                for f in feature:
                    if f in transaction:
                        cont = 1
                    else:
                        cont = 0
                feature_vector.append(cont)
            output_features.append(feature_vector)
        return np.matrix(output_features)
                                    
def oversample(instances,labels,oversampling_rate=2):

    instances = np.repeat(instances, oversampling_rate, axis=0)
    labels = np.repeat(labels, oversampling_rate, axis=0)
    instances = np.vstack(*instances)
    labels = np.vstack(*labels)
    return instances,labels

def sample_neighborhood_v1(node_name,order=2):

    node_container = queue.Queue(maxsize=200000000)
    node_container.put(node_name)
    neighborhoods = defaultdict(list)
    words = []
    for x in range(order):
        tmp_nbrs = []
        while not node_container.empty():
            nod = node_container.get()
            if x > 1:
                cpx = 1/np.exp(x)
            else:
                cpx = 1
            rn = np.random.uniform()
            if  rn < cpx:
                neighs = list(network.neighbors(nod))
                tmp_nbrs.append(neighs)
        for neighborhood in tmp_nbrs:
            ## word construction phase
            for node in neighborhood:                
                if args.word_type == "type_id_len":
                    word = node[1]+"_"+str(node[0])+"_"+str(x)                    
                elif args.word_type == "id_len":
                    word = str(node[0])+"_"+str(x)              
                elif args.word_type  == "type_len":
                    word = str(node[1])+"_"+str(x)
                elif args.word_type  == "type":
                    word = str(node[1])
                words.append(word)
                node_container.put(node)
    return words

def walk_kernel(ix,enx):
    node_container = queue.Queue(maxsize=200000000)
    node_container.put(start_node)
    tmp_walk = [] # [node_name]
    while not node_container.empty():
        nod = node_container.get()
        neighs = list(network.neighbors(nod))
        tar = random.choice(neighs)
        node_container.put(tar)
        if len(tmp_walk) > enx+1:
            break
        tmp_walk.append(tar)
    
    if args.word_type == "type_id_len":
        tmp_walk = "||".join([x[1]+"_"+str(x[0])+"_"+str(enx+1) for x in tmp_walk])
    elif args.word_type == "id_len":
        tmp_walk = "||".join([str(x[0])+"_"+str(enx+1) for x in tmp_walk])
    elif args.word_type  == "type_len":
        tmp_walk = "||".join([x[1]+"_"+str(enx+1) for x in tmp_walk])
    elif args.word_type  == "type":
        tmp_walk = "||".join([x[1] for x in tmp_walk])
    elif args.word_type  == "id":
        tmp_walk = "||".join([str(x[0]) for x in tmp_walk])
    return tmp_walk
    
def sample_neighborhood_v2(node_name,order=None,num_samples=10000,sampling_dist = "moyal"):

    """
    Main sampling routine..
    :param:
    node_name: name of the target node
    order: max walk len
    num_samples: num samples of a given wlen
    sampling dist: distribution of walk len samples
    """

    global start_node
    start_node = node_name
    walk_struct = []
    if sampling_dist == "moyal":
        r = moyal.rvs(size=num_samples)
        inds = np.histogram(r, order)
    elif sampling_dist == "levy":
        r = moyal.rvs(size=num_samples)
        inds = np.histogram(r, order)
    elif sampling_dist == "uniform":
        r = np.random.uniform(0,1,num_samples)
        inds = np.histogram(r, order)
    else:
        pass
    wlen_dist = inds[0]
    for enx, wlen in enumerate(wlen_dist):
        for j in range(wlen):
             tmp_walk = walk_kernel(node_name,enx)
             walk_struct.append(tmp_walk)
    return walk_struct

#@jit(void(int64[:],int64[:],int64[:],int32,int32,int32), nopython=True, nogil=True)
@jit(parallel=True,nogil=True,nopython=True)
def numba_walk_kernel(walk_matrix,sparse_pointers,sparse_neighbors,node_name,num_steps=3,num_walks=100):

    offset = 0
    num_neighs = 0
    for walk in prange(num_walks):
        curr = node_name
        offset = walk * (num_steps + 1)
        walk_matrix[offset] = node_name
        for step in prange(num_steps):
            num_neighs = sparse_pointers[curr+1] - sparse_pointers[curr]
            if num_neighs > 0:
                curr = sparse_neighbors[sparse_pointers[curr] + np.random.randint(num_neighs)]
            idx = offset+step+1
            walk_matrix[idx] = curr

def sample_neighborhood_v3(node_name,num_steps=3,num_walks=1000,sampling_dist = "moyal"):
    """
    Very fast node sampling.
    """

    walk_struct = []
    if sampling_dist == "moyal":
        r = moyal.rvs(size=num_walks)
        inds = np.histogram(r, num_steps)
    elif sampling_dist == "levy":
        r = moyal.rvs(size=num_walks)
        inds = np.histogram(r, num_steps)
    elif sampling_dist == "uniform":
        r = np.random.uniform(0,1,num_walks)
        inds = np.histogram(r, num_steps)
    else:
        pass
    wlen_dist = inds[0]
    for enx, wlen in enumerate(wlen_dist):
        walk_matrix = -np.ones((wlen, (enx+1)), dtype=np.int32, order='C')
        walk_matrix = np.reshape(walk_matrix, (walk_matrix.size,), order='C')
        numba_walk_kernel(walk_matrix,sparse_pointers,sparse_neighbors,node_name,num_steps=enx,num_walks=wlen)
        walk_mat = np.reshape(walk_matrix,(wlen,(enx+1)))[:,1:].flatten()
        walk_string = [str(x)+"_"+str(enx) for x in walk_matrix]
        walk_struct.append(walk_string)
    return walk_struct
    

def return_hmatrix_labels(net,max_samples=100):

    """
    Propositionalization of the relational database.
    """

    global network    
    global sparse_network
    global sparse_pointers
    global sparse_neighbors
    global target_node_map
    
    network = net
    target_node_map = {}
    sparse_network = nx.to_scipy_sparse_matrix(net).tocsr()
    sparse_pointers = sparse_network.indptr
    sparse_neighbors = sparse_network.indices
    
    target_nodes = []
    unique_types = []
    ntypes = {}
    label_vector = []
    for enx,node in enumerate(network.nodes(data=True)):
        target_node_map[node[0]] = enx
        if node[0][1] == args.target_label_tag:
            target_nodes.append((node[0],node[1]['labels']))
    final_documents = []
    
    for node in tqdm.tqdm(target_nodes):
        label_vector.append(int(node[1]))
        
        if args.sampler == "v3":
            ## distsampler -- compiled
            node_words = sample_neighborhood_v3(target_node_map[node[0]],args.order,args.num_samples,sampling_dist=args.sampling_distribution)
            node_words = itertools.chain(*node_words)            
            
        elif args.sampler == "v2":
            ## distsampler -- naive
            node_words = sample_neighborhood_v2(node[0],args.order,args.num_samples,sampling_dist=args.sampling_distribution)

        elif args.sampler == "v1":
            ## BFS
            node_words = sample_neighborhood_v1(node[0],args.order)

        if args.vectorizer == "binary":
            final_documents.append(" ".join(set(node_words))) ## counts do not matter here.
            
        elif args.vectorizer == "rules":
            final_documents.append(set(node_words))
            
        else:
            final_documents.append(" ".join(node_words))

    ## standard vectorizers
    if args.vectorizer == "binary":
        vec = CountVectorizer(ngram_range=(1,args.relation_order),max_features=args.num_features,binary=True)

    elif args.vectorizer == "hash":
        vec = HashingVectorizer(ngram_range=(1,args.relation_order),n_features=args.num_features)

    elif args.vectorizer == "tfidf":
        vec = TfidfVectorizer(ngram_range=(1,args.relation_order),max_features=args.num_features)
    elif args.vectorizer == "count":        
        vec = CountVectorizer(ngram_range=(1,args.relation_order),max_features=args.num_features,binary=False)
        
    elif args.vectorizer == "rules":
        vec = GarVectorizer(num_features=args.num_features,min_support=3)

    transformed_corpus = vec.fit_transform(final_documents)        
    label_matrix = np.array(label_vector)
    logging.info("Generated {} features.".format(transformed_corpus.shape[1]))
    
    assert len(label_matrix) == transformed_corpus.shape[0]
    return label_matrix, transformed_corpus, vec
    
if __name__ == "__main__":

    import argparse
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname",default="net_aminer.gpickle")
    parser.add_argument("--propositionalization",default=True)
    parser.add_argument("--target_label_tag",default="labeled_conference")
    parser.add_argument("--num_features",default=2000,type=int)
    parser.add_argument("--learner",default="LR")
    parser.add_argument("--random_seed",default=124876,type=int)
    parser.add_argument("--vectorizer",default="binary")
    parser.add_argument("--word_type",default="id_len")
    parser.add_argument("--sampler",default="v3")
    parser.add_argument("--relation_order",default=3,type=int)
    parser.add_argument("--order",default=10,type=int)
    parser.add_argument("--sampling_distribution",default="uniform",type=str)
    parser.add_argument("--num_samples",default=10000,type=int)
    args = parser.parse_args()
    outname = args.target_label_tag+"_"+args.fname.split("/")[-1].replace("gpickle","mat")
    
    if args.propositionalization:
        multilayer_network = nx.read_gpickle(args.fname)
        print(nx.info(multilayer_network))
        label_matrix, feature_matrix, vectorizer= return_hmatrix_labels(multilayer_network)
        spmat = sparse.csr_matrix(feature_matrix)        

        representation = umap.UMAP(n_neighbors=5, min_dist=0.05).fit_transform(spmat)
        g = sns.scatterplot(representation[:,0],representation[:,1],hue=label_matrix,style=label_matrix,palette="Set2", legend=False,s = 100)
        plt.axis('off')
        plt.show()
