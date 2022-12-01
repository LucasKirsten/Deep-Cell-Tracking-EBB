# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:15:12 2020
@author/: crjun
"""

import cv2
import rtree.index
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon


def rect_polygon(ptslist):
    """Return a shapely Polygon describing the rectangle with centre at
    (x, y) and the given width and height, rotated by angle quarter-turns.
    """
    return Polygon(ptslist.T)

def rotated_iou(box1, box2):
    # from (xc,yc),(w,h),angle to points
    pts1 = cv2.boxPoints(((box1[0],box1[1]),(box1[2],box1[3]),box1[4]))
    pts2 = cv2.boxPoints(((box2[0],box2[1]),(box2[2],box2[3]),box2[4]))
    
    P1 = Polygon(list(pts1))
    P2 = Polygon(list(pts2))
    inter = P1.intersection(P2).area
    union = P1.union(P2).area
    return inter/union


def intersection_over_union(rects_a, rects_b):
    """Calculate the intersection-over-union for every pair of rectangles
    in the two arrays.
    Arguments:
    rects_a: array_like, shape=(M, 5)
    rects_b: array_like, shape=(N, 5)
        Rotated rectangles, represented as (centre x, centre y, width,
        height, rotation in quarter-turns).
    Returns:int
    iou: array, shape=(M, N)
        Array whose element i, j is the intersection-over-union
        measure for rects_a[i] and rects_b[j].
    """
    m = len(rects_a)
    n = len(rects_b)
    if m > n:
        # More memory-efficient to compute it the other way round and
        # transpose.
        return intersection_over_union(rects_b, rects_a).T
    
    # Convert rects_a to shapely Polygon objects.
    polys_a = [rect_polygon(*r) for r in rects_a]

    # Build a spatial index for rects_a.
    index_a = rtree.index.Index()
    for i, a in enumerate(polys_a):
        index_a.insert(i, a.bounds)

    # Find candidate intersections using the spatial index.
    iou = np.zeros((m, n))
    for j, rect_b in enumerate(rects_b):
        b = rect_polygon(*rect_b)
        for i in index_a.intersection(b.bounds):
            a = polys_a[i]
            intersection_area = a.intersection(b).area
            if intersection_area:
                iou[i, j] = intersection_area / a.union(b).area

    return iou