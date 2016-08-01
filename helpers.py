import pandas as pd
import numpy as np

def test_1(x, y):
    return x, y

def ingrp(df, label, grp):
    ingrp_label = label + '_ingrp'

    if not isinstance(grp, list):
        grp = [grp]

    df[ingrp_label] = False
    df.loc[df[label].isin(grp), ingrp_label] = True

    return df

def isblank(df, label):
    isblank_label = label + '_isblank'

    df[isblank_label] = False
    df = df.replace(r'\s+|^$', np.nan, regex=True)
    df.loc[df[label].isnull(), isblank_label] = True

    return df
