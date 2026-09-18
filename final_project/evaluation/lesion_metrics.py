import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def compute_lesion_metrics(pred_mask, gt_mask, tolerance=15.0):
    """
    Computes lesion-level one-to-one matching metrics based on centroid distances.
    pred_mask: 2D binary numpy array [H, W]
    gt_mask: 2D binary numpy array [H, W]
    tolerance: Max distance in pixels to consider a match valid.
    """
    # 1. Extract connected components
    num_gt, _, stats_gt, centroids_gt = cv2.connectedComponentsWithStats(gt_mask.astype(np.uint8), connectivity=8)
    num_pd, _, stats_pd, centroids_pd = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
    
    # Remove background (label 0)
    centroids_gt = centroids_gt[1:]
    centroids_pd = centroids_pd[1:]
    
    n_gt = len(centroids_gt)
    n_pd = len(centroids_pd)
    
    if n_gt == 0 and n_pd == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0, 'pred_count': 0}
        
    if n_gt == 0:
        return {'tp': 0, 'fp': n_pd, 'fn': 0, 'gt_count': 0, 'pred_count': n_pd}
        
    if n_pd == 0:
        return {'tp': 0, 'fp': 0, 'fn': n_gt, 'gt_count': n_gt, 'pred_count': 0}
        
    # 2. Compute distance matrix
    # dist_matrix shape: [n_gt, n_pd]
    dist_matrix = cdist(centroids_gt, centroids_pd, metric='euclidean')
    
    # 3. Hungarian algorithm for optimal bipartite matching
    # We want to minimize distance
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # 4. Filter matches by tolerance
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= tolerance:
            tp += 1
            
    fp = n_pd - tp
    fn = n_gt - tp
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'gt_count': n_gt,
        'pred_count': n_pd
    }
