"""
text recognition eval

"""

def eval_text(preds:[str],gts:[str])->dict:
    assert len(preds)==len(gts),"eval the preds and gts must be the same length"
    length = len(preds)
    norm_ED = 0
    n_correct = 0
    n_incorrect = 0
    for pred,gt in zip(preds,gts):
        if pred==gt:
            n_correct += 1
        else:
            n_incorrect +=1

        if len(gt) == 0 or len(pred) == 0:
            norm_ED += 0
        elif len(gt) > len(pred):
            norm_ED += 1 - edit_distance(pred, gt) / len(gt)
        else:
            norm_ED += 1 - edit_distance(pred, gt) / len(pred)


    accuracy = n_correct / float(length)
    norm_ED = norm_ED / float(length)  # ICDAR2019 Normalized Edit Distance

    recall = n_correct / float(length)
    f1 = 0 if (accuracy + recall) == 0 else (2.0 * accuracy * recall) / (accuracy + recall)
    return dict(
        acc=accuracy,
        recall=recall,
        f1=f1,
        normed=norm_ED
    )










            # '''
            # (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            # "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            # if len(gt) == 0:
            #     norm_ED += 1
            # else:
            #     norm_ED += edit_distance(pred, gt) / len(gt)
            # '''
            #
            # # ICDAR2019 Normalized Edit Distance
            # if len(gt) == 0 or len(pred) == 0:
            #     norm_ED += 0
            # elif len(gt) > len(pred):
            #     norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            # else:
            #     norm_ED += 1 - edit_distance(pred, gt) / len(pred)



"""
editDistance from nltk 
# -*- coding: utf-8 -*-
# Natural Language Toolkit: Distance Metrics
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Tom Lippincott <tom@cs.columbia.edu>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
#
"""
def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to
    substitutions.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                substitution_cost=substitution_cost,
                transpositions=transpositions,
            )
    return lev[len1][len2]
