from typing import Any

import pandas as pd
import numpy as np


MAX_WIDTH = 20
ELLIPSIS = ' ...'
ADJ_WIDTH = MAX_WIDTH - len(ELLIPSIS)


def _truncate(s: Any) -> str:
    s = str(s)
    if len(s) >= ADJ_WIDTH:
        return s[:ADJ_WIDTH] + ELLIPSIS
    return s


def df_to_docstring(df: pd.DataFrame) -> str:
    docstring = ''
    widths = []
    index_width = len(str(len(df) - 1)) 

    df = df.replace({np.nan: 'NaN'}).replace({pd.NaT: 'NaT'})
    for col in df:
        df[col] = df[col].astype(str)
        widths.append(min(max([df[col].str.len().max(), len(col)]), MAX_WIDTH))
        
    docstring += ' '*index_width
    hline = '-'*index_width
    
    for width, col in zip(widths, df):
        docstring += f' | {_truncate(col): <{width}}'
        hline += f'-|-{"-"*width}'

    docstring += '\n' + hline + '\n'

    for i, row in df.iterrows():
        docstring += f'{i: >{index_width}}'

        for width, val in zip(widths, row):
            docstring += f' | {_truncate(val): >{width}}'
        docstring += '\n'
    
    return docstring
