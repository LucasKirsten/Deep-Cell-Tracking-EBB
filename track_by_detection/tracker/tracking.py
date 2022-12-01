# -*- coding: utf-8 -*-
"""
Code for solving the Bise et al algorithm
"RELIABLE CELL TRACKING BY GLOBAL DATA ASSOCIATION"

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import cvxpy
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .func_utils import *
from .classes import Tracklet

#%% make hyphotesis
def _make_hypotheses(tracklets, Nf, false_positive):
    
    Nx = len(tracklets)
    SIZE = 2*Nx+Nf
    
    # define generator for hyphoteses
    for k in range(Nx):
        
        track = tracklets[k]
        
        # intialize variables
        hyp,C,pho = [],[],[]
            
        # determine alpha value based on normal and mitoses precision
        num_mit = np.sum([int(det.mit) for det in track])
        num_normal = len(track) - num_mit
        alpha = (num_normal*ALPHA_NORMAL + num_mit*ALPHA_MITOSE)/len(track)
        
        # false/true positive hypothesis
        if false_positive and track.score()<TRACK_SCORE_TH:
            # false positive
            Ch = np.zeros(SIZE, dtype='bool')
            Ch[[i==k or i==Nx+k for i in range(SIZE)]] = 1
            ph = PFP(track, alpha)
            
            hyp.append(f'fp_{track.idx}')
            C.append(Ch)
            pho.append(ph)
            
            # true positive
            ph = PTP(track, alpha)
            
            hyp.append(f'tp_{track.idx}')
            C.append(Ch)
            pho.append(ph)
        
        k2 = k
        for track2 in tracklets[k+1:]:
            k2 += 1
            
            # translation hyphotesis
            # if tracklets begins togheter or is not possible to join
            if track.end<track2.start or track.end+len(track2)<=Nf:
            
                # calculate center distances
                cnt_dist = center_distances(track[-1].cx, track[-1].cy,
                                            track2[0].cx, track2[0].cy)
                
                # verify hyphotesis
                if track2.start-track.end<LINK_TH and \
                    cnt_dist<CENTER_TH and track.end<track2.start:
                    
                    # translation hypothesis
                    Ch = np.zeros(SIZE, dtype='bool')
                    Ch[[i==k or i==Nx+k2 for i in range(SIZE)]] = 1
                    ph = Plink(track, track2, cnt_dist)
                    
                    hyp.append(f'transl_{track.idx},{track2.idx}')
                    C.append(Ch)
                    pho.append(ph)
                
            # iterate over the mitoses events
            k3 = k2
            for track3 in tracklets[k2+1:]:
                k3 += 1
                
                # if tracklets begins togheter or is above gap threshold
                if track.end>=track2.start or track.end>=track3.start:
                    continue
                
                if track2.start-track.end>MIT_TH or track3.start-track.end>MIT_TH:
                    continue
                
                cnt_dist2 = center_distances(track[-1].cx, track[-1].cy,
                                             track2[0].cx, track2[0].cy)
                if cnt_dist2>CENTER_MIT_TH:
                    continue
                cnt_dist3 = center_distances(track[-1].cx, track[-1].cy,
                                             track3[0].cx, track3[0].cy)
                if cnt_dist3>CENTER_MIT_TH:
                    continue
                cnt_dist23 = center_distances(track2[0].cx, track2[0].cy,
                                              track3[0].cx, track3[0].cy)
                if cnt_dist23>CENTER_MIT_TH:
                    continue
                
                # mitoses hypothesis
                Ch = np.zeros(SIZE, dtype='bool')
                Ch[[i==k or i==Nx+k2 or i==Nx+k3 for i in range(SIZE)]] = 1
                d_mit = (track2.start-track.end + track3.start-track.end)/2
                cnt_dist = (cnt_dist2+cnt_dist3)/2
                ph = Pmit(cnt_dist, d_mit)
                
                hyp.append(f'mit_{track.idx},{track2.idx},{track3.idx}')
                C.append(Ch)
                pho.append(ph)
        
        yield hyp,C,pho

#%% adjust the tracklets to the frames

def _adjust_tracklets(tracklets, hyphotesis, add_parent):
    
    # sort hyphotesis based on the tracklet position
    hyphotesis = sorted(hyphotesis,\
                        key=lambda x:int(x.split('_')[-1].split(',')[0].split('-')[0]))
    
    # make list of tracklets into a dict of their indexes
    tracklets = {int(track.idx):track for track in tracklets}
    
    # adjust tracklets for each hyphotesis
    for hyp in hyphotesis:
        
        # verify the hyphotesis name and its values
        mode,idxs = hyp.split('_')
        
        if 'fp' in mode:
            indexes = idxs.split(',')
            tracklets.pop(int(idxs))
                
        elif 'transl' in mode:
            idx1,idx2 = idxs.split(',')
            track = tracklets.pop(int(idx1))
            track.join(tracklets[int(idx2)])
            track.set_idx(int(idx2))
            tracklets[int(idx2)] = track
            
        elif 'mit' in mode and add_parent:
            idx1,idx2,idx3 = idxs.split(',')
            tracklets[int(idx2)].parent = tracklets[int(idx1)]
            tracklets[int(idx3)].parent = tracklets[int(idx1)]
            
    return list(tracklets.values())

#%% solve integer optimization problem

def _solve_optimization(C, pho, hyp):
    # solve integer optimization problem
    
    x = cvxpy.Variable((len(pho),1), boolean=True)
    
    # define and solve integer optimization
    constrains = [(cvxpy.transpose(C) @ x) <= 1]
    total_prob = cvxpy.sum(cvxpy.transpose(pho) @ x)
    
    if DEBUG: print('Solving integer optimization...')
    knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_prob), constrains)
    knapsack_problem.solve(solver=cvxpy.CBC) #cvxpy.GLPK_MI
    
    # get the true hypothesis
    x = np.squeeze(x.value).astype('int')
    hyp = np.array(hyp)[x==1]
    
    return hyp

#%% populate C and pho matrixes for hypothesis
def _get_C_pho_matrixes(tracklets, Nf, false_positive):
    
    hyp,C,pho = [],[],[]
    def _populate_matrixes(h,c,p):
        hyp.extend(h)
        C.extend(c)
        pho.extend(p)
        
    pbar = _make_hypotheses(tracklets, Nf, false_positive)
    if DEBUG:
        pbar = tqdm(pbar, total=len(tracklets))
        pbar.set_description('Bulding hyphotesis matrix: ')
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        _ = parallel(delayed(_populate_matrixes)(h,c,p) for h,c,p in pbar)
    
    return np.array(C), np.float32(pho), np.array(hyp)

#%%

def solve_tracklets(tracklets:list, Nf:int) -> list:
    '''
    
    Solve the tracklets using the Bise et al algorithm.

    Parameters
    ----------
    tracklets : list
        List of Tracklets.
    Nf : int
        Total number of frames.

    Returns
    -------
    list
        List of solved Tracklets.

    '''
    
    # solve iteratively
    # last_track_size = len(tracklets)
    # for it in range(max_iterations):
    #     if DEBUG: print(f'Iteration {it+1}/{max_iterations}:')
        
    #     tracklets = np.copy(tracklets)
        
    #     # get hypothesis matrixes
    #     C, pho, hyp = _get_C_pho_matrixes(tracklets, Nf)
            
    #     # solve integer optimization
    #     hyp = _solve_optimization(C, pho, hyp)
        
    #     if DEBUG: print('Adjusting final tracklets...')
    #     adj_tracklets = _adjust_tracklets(tracklets, hyp)
        
    #     # verify if to stop iterations
    #     current_track_size = len(adj_tracklets)
    #     if current_track_size==last_track_size or it+1==max_iterations:
    #         if DEBUG: print('Early stop.')
    #         break
        
    #     # update variables for next iterations and remove parentins
    #     last_track_size = current_track_size
    #     def __remove_parent(track):
    #         track.parent = None
    #         return track
    #     tracklets = [__remove_parent(track) for track in adj_tracklets]
    #     if DEBUG: print()
        
    # solve hypothesis
    if DEBUG: print('Solving...')
        # solve for translation and mitoses
    C, pho, hyp = _get_C_pho_matrixes(tracklets, Nf, False)
    hyp = _solve_optimization(C, pho, hyp)
    tracklets = _adjust_tracklets(tracklets, hyp, False)
    
        # solve for false positives
    C, pho, hyp = _get_C_pho_matrixes(tracklets, Nf, True)
    hyp = _solve_optimization(C, pho, hyp)
    tracklets = _adjust_tracklets(tracklets, hyp, True)
    
    # remove empty tracklets
    tracklets = [t for t in tracklets if len(t)>0]
        
    # set idx values
    tracklets = sorted(tracklets, key=lambda x:x.start)
    for i,track in enumerate(tracklets):
        track.set_idx(i+1)
        
    return tracklets






