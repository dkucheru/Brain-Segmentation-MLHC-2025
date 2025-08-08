import sys
sys.path.insert(0, '../..')

import numpy as np
import cv2
import math
from operator import itemgetter
import constants.global_constants as gc

surface_distance = __import__('surface-distance.surface_distance')


def delete_outs(preds):
    summation = np.sum(preds.reshape(preds.shape[0], -1), axis=-1)

    for slice_index in range(summation.shape[0]):
        prev_slice = -1
        next_slice = -1

        curr = summation[slice_index]
        if slice_index != 0 or slice_index % preds.shape[1] != 0:
            prev_slice = summation[slice_index - 1]
        if slice_index + 1 != summation.shape[0]:
            next_slice = summation[slice_index + 1]

        if curr >= 0 and (prev_slice < 2 and next_slice < 2):
            preds[slice_index] = 0
            summation[slice_index] = 0

    return preds


def dice_coef(true_y, pred_y):
    ty = true_y.ravel()
    py = pred_y.ravel()
    return (2. * np.sum(ty * py) + 1) / (np.sum(ty) + np.sum(py) + 1)


def hausdorff_coef(t, pred):
    dists = surface_distance.compute_surface_distances(t, pred, spacing_mm=(1, 1, 1))
    return surface_distance.compute_robust_hausdorff(dists, 100)


def disparity(t, pred):
    return (np.sum(pred) - np.sum(t)) / np.sum(t)


def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def contour_approximation(mask, minimum_distance_centroid=None, orig=None):
    if orig is None:
        orig = mask
    full_mask = np.sum(orig, axis=0)
    full_mask[full_mask > 0] = 1
    full_mask = full_mask.astype(np.uint8)
    contours, nd = cv2.findContours(full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    error_blobs = []
    for cont in contours:
        M = cv2.moments(cont)
        if M['m00'] == 0:
            continue
        centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        error_blobs.append((cont, centroid, cv2.contourArea(cont)))

    error_blobs = sorted(error_blobs, key=itemgetter(2))
    error_blobs.reverse()

    if len(error_blobs) == 0:
        return mask

    mask_uint = mask.astype(np.uint8)
    big_blob = error_blobs[0]
    blob_map = np.zeros(mask_uint.shape[1:], dtype='uint8')
    cv2.drawContours(blob_map, [big_blob[0]], -1, 1, -1)

    for ind in range(mask_uint.shape[0]):
        mask_uint[ind] *= blob_map.astype(np.uint8)

        if minimum_distance_centroid is not None:
            contours, nd = cv2.findContours(mask_uint[ind], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            error_blobs = []
            for cont in contours:
                M = cv2.moments(cont)
                if M['m00'] == 0:
                    continue
                error_blobs.append((cont, (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])), cv2.contourArea(cont)))

            if len(error_blobs) < 2:
                continue

            for cont, centroid, area in error_blobs:
                if distance(centroid, big_blob[1]) >= minimum_distance_centroid:
                    cv2.drawContours(mask_uint[ind], [cont], -1, 0, -1)

    return mask_uint.astype(np.int)

