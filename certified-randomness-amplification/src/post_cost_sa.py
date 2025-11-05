###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import copy
import numpy as np
from tqdm import tqdm
import random

def SA_kalachev(tree,
                tinit = 10,
                tfin = 1e-4,
                tsteps = 1e2,
                inplace = False,
                progress = False,
                pre_weight = 1):

    
    tree = tree if inplace else copy.deepcopy(tree)
    root = tree.all_tens
    cost = tree.total_cost()
    post_cost = tree.post_cost()
    
    temps = np.linspace(tinit,tfin,int(tsteps))
    
    costs = []
    post_costs = []
    

    pbar = tqdm(temps, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', disable = not progress)
    
    
    
    for temp in pbar:
        
        
        dat_sweep = dict()        
        beta = 1/temp
        parents = [root]
        while parents:

            p = parents.pop(0)
            l,r = tree.chil[p]

            if len(l) == 1 and len(r) == 1:

                # Both are leaves
                continue

            elif len(r) == 1:

                # only r is leaf
                Y,c = l,r

            elif len(l) ==1:

                # only l is leaf
                Y,c = r,l

            else:

                # neither is leaf
                
                s = random.getrandbits(1)
                Y,c = tree.chil[p][s], tree.chil[p][1-s]


            #Now randomly order the children of Y
            
            s = random.getrandbits(1)
            a,b = tree.chil[Y][s],tree.chil[Y][1-s]

            # compute part of old cost subject to change:
            old_post_cost = tree.cost[p] * (1 if tree.post[p] else 0) + tree.cost[Y] * (1 if tree.post[Y] else 0)
            old_cost = tree.cost[p] + tree.cost[Y]

            # compute new node structure and part of new cost subject to change:

            new_Y = a | c
            new_c = b
            new_a = a
            new_b = c

            new_Y_inds = {a:tree.inds[Y][a], c:tree.inds[p][c]}
            new_Y_cost = 2**len(new_Y_inds[a] | new_Y_inds[c])

            
            new_Y_post = True if (new_Y & tree.post_tensors) else False     
            new_p_inds = {new_Y: tree.external_inds_union(new_Y_inds), b:tree.inds[Y][b]}
            new_p_cost = 2**len(new_p_inds[new_Y] | new_p_inds[b])
            
            
            new_post_cost = new_p_cost * (1 if tree.post[p] else 0) + new_Y_cost * (1 if new_Y_post else 0)
            new_cost = new_p_cost + new_Y_cost
            
            dE = float((new_post_cost + pre_weight*(new_cost-new_post_cost)) - (old_post_cost + pre_weight*(old_cost-old_post_cost)))
            if dE <= 0:
                accept = True
            elif np.log(np.random.rand()) < - (beta * np.log2(dE)):
                accept = True
            else:
                accept = False
                

            if accept:

                tree.chil.pop(Y)
                tree.cost.pop(Y)
                tree.inds.pop(Y)
                tree.post.pop(Y)
                
                tree.chil[p] = [new_Y,new_c]
                tree.chil[new_Y] = [new_a,new_b]

                tree.inds[p] = new_p_inds
                tree.cost[p] = new_p_cost
            
                tree.inds[new_Y] = new_Y_inds
                tree.cost[new_Y] = new_Y_cost
                tree.post[new_Y] = new_Y_post
                
                cost = cost + (new_cost - old_cost)
                post_cost = post_cost + (new_post_cost - old_post_cost)
                
                if post_cost < 0:
                    print(f'({new_post_cost},{old_post_cost})')
               
            l,r = tree.chil[p]

            if len(l) > 2:
                parents.append(l)
            if len(r) > 2:
                parents.append(r)
            
            
            dat_sweep[post_cost] = cost

        min_ps = min(dat_sweep.keys())
        post_costs.append(min_ps)
        costs.append(dat_sweep[min_ps])
        
        
    return costs, post_costs