###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import copy
import random


def tn_from_ctg_tree(tree):
    
    ctg_info = tree.inputs
    tn_data = dict()

    for j in range(len(ctg_info)):

        tn_data[j] = set(ctg_info[j])

    
    all_inds = set.union(*list(tn_data.values()))
    ind_dict = {ind:j for j,ind in enumerate(list(all_inds))}
    
    tensors = {tn:frozenset([ind_dict[_] for _ in tn_data[tn]]) for tn in tn_data}
   
    
    return tensors

def tn_from_quimb_amp(tn):
        
    tn_data = dict()
    post_tensors = []
    
    for j,t in enumerate(tn.tensors):
        tn_data[j] = set(t.inds)
        if 'post tensor' in t.tags: post_tensors.append(j)

    all_inds = set.union(*list(tn_data.values()))
    ind_dict = {ind:j for j,ind in enumerate(list(all_inds))}
    tensors = {tn:frozenset([ind_dict[_] for _ in ind_set]) for tn,ind_set in tn_data.items()}
    
    return tensors, frozenset(post_tensors)



def tn_from_quimb_psi(psi):
    
    
    def strip_ks(t):

            inds = t.inds
            return {ind for ind in inds if ind[0]!='k'}

    def get_delayed_inds(t):

        status = False
        inds = t.inds

        for ind in inds:
            if ind[0] == 'k':
                status = True

        if status:
            remaining_inds = {ind for ind in inds if ind[0]!='k'}
            return remaining_inds
        else:
            return set()
        
    tn_data = dict()
    delayed_inds = set()

    for j in range(len(psi.tensors)):

        t = psi.tensors[j]
        delayed_inds = delayed_inds.union(get_delayed_inds(t))
        tn_data[j] = strip_ks(t)

    
    delayed_tns = [tn for tn in tn_data if tn_data[tn] & delayed_inds]
    
    all_inds = set.union(*list(tn_data.values()))
    ind_dict = {ind:j for j,ind in enumerate(list(all_inds))}
    
    tensors = {tn:frozenset([ind_dict[_] for _ in tn_data[tn]]) for tn in tn_data}
    post_tensors = frozenset(delayed_tns)
    
    return tensors, post_tensors


class ContractionTree:

    def __init__(self, tensors, post_tensors = 'all'):
        
        self.tensors = tensors
        self.all_tens = frozenset(tensors.keys())
        self.all_inds = frozenset([_ for t in tensors for _ in tensors[t]])
        self.indices = {idx: frozenset([tn for tn,inds in self.tensors.items() if idx in inds]) for idx in self.all_inds}

        
        self.chil = dict() # Dictionary mapping nodes to children, defines contraction tree
        self.inds = dict() # Dictionary mapping nodes to the a list of the childrens external indices
        self.cost = dict() # Dictionary mapping nodes to costs associated with absorbing children into parents
        self.post = dict() # Dictionary mapping nodes to bool for whether or not to include cost
              
        
        if post_tensors == 'all':
            
            self.post_tensors = self.all_tens
        
        else:
            
            self.post_tensors = post_tensors
        
        self.autotree_ballanced(self.all_tens)
        self.assign_node_properties()


    def add_node(self,s,l,r):

        self.chil[frozenset(s)] = [frozenset(l),frozenset(r)]                
    
    def autoseed_ballanced(self,nodes):

        children = []
        rnodes = copy.copy(list(nodes))
        random.shuffle(rnodes)

        l = len(rnodes)
        children.append(frozenset(rnodes[:l//2]))
        children.append(frozenset(rnodes[l//2:l]))

        self.add_node(rnodes,children[0],children[1])
    
    def autobranch_ballanced(self, return_progress = False):
        
        current_tree = copy.copy(self.chil)
        if return_progress:
            progress = False

        for node in current_tree:
            ca,cb = current_tree[node]
            for child in ca,cb:         
                if len(child)>1 and child not in current_tree:
                    if return_progress:
                        progress = True
                    s = list(child)
                    random.shuffle(s)
                    length = len(s)
                    l = s[:length//2]
                    r = s[length//2:]
                    self.add_node(s,l,r)

        if return_progress:
            return progress
            
    
    def autotree_ballanced(self,nodes):

        self.autoseed_ballanced(nodes)                    
        status = True
        while status:
            status = self.autobranch_ballanced(return_progress = True)
            
    def from_ctg_tree(self,ctg_tree):
        
        self.chil = ctg_tree.children
        self.assign_node_properties()
    
    def get_external_inds(self, children):
        
        data = dict()    
        for c in children:
            
            inds = frozenset.union(*list([self.tensors[t] for t in c]))
            data[c] = frozenset([idx for idx in inds if not self.indices[idx] <= c])
            
        return data
    
    def external_inds_union(self, info):
        
        # Meant to be a faster version of get_external_inds that benefits from already knowing the external inds of
        # two sets and computing the external inds of their merger frugally.
        
        # info should be a dictionary like:
        # {s1:{external inds of set s1}, s2:{external inds of set s2}}
        
        s1,s2 = info.keys()
        pool_tns = s1.union(s2)
        pool_ext_inds = info[s1].union(info[s2])
        
        obvious_ext_inds = info[s1] ^ info[s2]
        maybe_ext_inds = pool_ext_inds - obvious_ext_inds
        valid_ext_inds = frozenset([idx for idx in maybe_ext_inds if not self.indices[idx] <= pool_tns])
        ext_inds = obvious_ext_inds.union(valid_ext_inds)
        
        return ext_inds
        

        
        
    def assign_node_properties(self):

        self.inds = dict()
        self.cost = dict()
        self.post = dict()

        for p in self.chil:

            c0,c1 = children = self.chil[p]
            data = self.get_external_inds(children)        
            self.inds[p] = data
            self.cost[p] = 2**len(data[c0] | data[c1])
            self.post[p] = True if (p & self.post_tensors) else False
            
    def update_cost(self,p):

        c0,c1 = children = self.chil[p]
        data = self.inds[p]
        self.cost[p] = 2**len(data[c0] | data[c1])
    
    def update_inds(self,p):
        
        l,r = self.chil[p]
        self.inds[p] = self.get_external_inds([l,r])
     
        
    def update_post(self,p):

        self.post[p] = True if p & self.post_tensors else False

    
    def total_cost(self):
        
        return sum(self.cost.values())
    
    def post_cost(self):
        
        return sum([self.cost[p]*(1 if self.post[p] else 0) for p in self.cost])


