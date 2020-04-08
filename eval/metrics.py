
import numpy as np


def get_map(run, R):
    """ Calculate mean average precision (MAP). """
    # Sum relevant docs in run.
    R_run = sum(run)
    # If relevant docs in run and total relevant docs above 0.
    if (R_run > 0.0) and (R > 0):
        # Initialise precision counter.
        precision_sum = 0
        # Loop of run and append precision to precision counter.
        for i, r in enumerate(run):
            if r == 1.0:
                precision_sum += get_precision(run=run, k=(i+1))
        # Divide precision counter by total relevant docs.
        return precision_sum / R
    else:
        return 0.0


def get_R_prec(run, R):
    """ Calculate R-precision. """
    if R > 0:
        # Reduce run at index #R and calculate precision.
        return sum(run[:R]) / R
    else:
        return 0.0


def get_recip_rank(run):
    """ Calculate reciprocal rank. """
    for i, r in enumerate(run):
        # Return 1 / rank for first relevant.
        if r == 1.0:
            return 1/(i+1)
    return 0.0


def get_precision(run, k=20):
    """ Calculate precision at kth rank. """
    run_k = run[:k]
    return sum(run_k) / k


def get_recall(run, R, k=40):
    """ Calculate recall at kth rank """
    run_k = run[:k]
    R_run = sum(run_k)
    if R > 0:
        return R_run / R
    return 1.0


def get_ndcg(run, R, k=20):
    """ Calculate normalised discount cumulative gain (NDCG) at kth rank. """
    run_k = run[:k]
    # Initialise discount cumulative gain.
    dcg = 0
    # Initialise perfect discount cumulative gain.
    i_dcg = 0
    R_run = sum(run)
    if (R_run > 0) and (R > 0):
        for i, r in enumerate(run_k):
            if i == 0:
                if (i+1) <= R:
                    i_dcg += 1
                dcg += r
            else:
                discount = np.log2(i+2)
                if (i+1) <= R:
                    i_dcg += 1 / discount
                dcg += r / discount
        # Normalise cumulative gain by dividing 'discount cumulative gain' by 'perfect discount cumulative gain'.
        return dcg / i_dcg
    else:
        return 0


