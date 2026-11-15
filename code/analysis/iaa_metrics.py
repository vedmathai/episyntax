"""
Inter-annotator agreement metrics used in the EpiSyntax human-validation
pilot: Krippendorff's alpha, Fleiss' kappa, Gwet's AC1, and
ICC(3,1)/ICC(3,k). Thin wrappers around well-known packages
(krippendorff, statsmodels, irrCAC, pingouin) rather than hand-rolled
implementations, verified during development to match a from-scratch
reference implementation on both synthetic and project data.
"""
import numpy as np
import pandas as pd
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa
from irrCAC.raw import CAC
import pingouin as pg


def krippendorff_alpha_ragged(units, metric='interval', categories=None):
    """
    units: list of per-unit rating lists whose length (number of raters)
    may vary, e.g. because some raters are excluded on some units. Units
    with fewer than 2 ratings carry no reliability information and are
    dropped by the underlying package.
    metric: 'nominal', 'ordinal', 'interval', or 'ratio'
        (krippendorff's level_of_measurement).
    categories: ignored for 'nominal'/'interval' (inferred from the data);
        kept for signature compatibility with callers that pass it.
    """
    max_raters = max(len(u) for u in units)
    reliability_data = np.full((max_raters, len(units)), np.nan)
    for i, vals in enumerate(units):
        reliability_data[:len(vals), i] = vals
    return krippendorff.alpha(reliability_data=reliability_data, level_of_measurement=metric)


def fleiss_kappa(matrix, categories):
    """matrix: 2D array-like, shape (n_units, m_raters), no missing
    values. categories: list of category labels."""
    matrix = np.asarray(matrix)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    table = np.zeros((matrix.shape[0], len(categories)))
    for u, row in enumerate(matrix):
        for v in row:
            table[u, cat_to_idx[v]] += 1
    return sm_fleiss_kappa(table, method='fleiss')


def gwet_ac1_ragged(units, categories):
    """
    Gwet's AC1 (Gwet 2008) -- like Fleiss' kappa/Krippendorff's alpha, but
    with a chance-agreement estimate that stays near zero when one category
    dominates, instead of ballooning toward the observed agreement rate.
    Fixes the "kappa/AC1 paradox": kappa/alpha assume raters who disagree
    are effectively guessing according to the category prevalence, so under
    extreme skew the model predicts high agreement by chance too, and the
    chance-corrected ratio destabilizes (can read near 0 or negative
    despite near-total raw agreement). AC1 instead assumes chance
    agreement is spread evenly across categories, which keeps the baseline
    low and the statistic close to raw agreement when prevalence is skewed.

    units: list of per-unit rating lists (ragged length allowed).
    categories: list of category labels.

    irrCAC's CAC class requires its ratings argument to be a pandas
    DataFrame (subjects x raters, NaN for missing) -- that's its actual
    API, not a choice made here. The DataFrame is built only at this call
    boundary; nothing else in this module touches pandas.
    """
    max_raters = max(len(u) for u in units)
    ratings = np.full((len(units), max_raters), np.nan)
    for i, vals in enumerate(units):
        ratings[i, :len(vals)] = vals
    # digits=15 (default is 5) -- avoid irrCAC's default rounding, which
    # would otherwise lose precision relative to the rest of this pipeline.
    cac = CAC(pd.DataFrame(ratings), categories=categories, digits=15)
    return cac.gwet()['est']['coefficient_value']


def icc_3(matrix):
    """ICC(3,1) and ICC(3,k) -- two-way mixed effects, fixed raters
    (Shrout & Fleiss 1979 / McGraw & Wong 1996), via pingouin.

    pingouin's intraclass_corr requires long-format data as a pandas
    DataFrame -- that's its actual API, not a choice made here. The
    DataFrame is built only at this call boundary; nothing else in this
    module touches pandas. pingouin labels these ICC(C,1) and ICC(C,k)
    ("consistency", fixed raters) rather than ICC(3,1)/ICC(3,k), but
    they're the same statistic.
    """
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    rows = [
        {'subject': subject, 'rater': rater, 'rating': matrix[subject, rater]}
        for subject in range(n) for rater in range(k)
    ]
    result = pg.intraclass_corr(data=pd.DataFrame(rows), targets='subject', raters='rater', ratings='rating')
    icc_31 = result.loc[result['Type'] == 'ICC(C,1)', 'ICC'].iloc[0]
    icc_3k = result.loc[result['Type'] == 'ICC(C,k)', 'ICC'].iloc[0]
    return icc_31, icc_3k
