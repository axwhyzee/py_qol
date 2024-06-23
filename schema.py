import copy
import logging
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Iterable
import warnings
import yaml

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(20)

warnings.filterwarnings('ignore')

# the following types are converted using custom logic instead of via mapping
# - object
# - datetime[ns]  \____ These 2 datetime fmts could contain date or timestamp info, 
# - datetime[us]  /     so we can't use a mapping directly
PANDAS_TO_ARROW_TYPES = {
    'float64': 'double',
    'int64': 'int64',
    'bool': 'bool'
}

# hierachy is established in terms of backwards compatability of PyArrow types
TYPE_HIERACHY = {
    'int64': 'double',
    'double': 'string',
    'bool': 'string',
    'date32[day]': 'timestamp[us, tz=UTC]',
    'timestamp[us, tz=UTC]': 'string'
}

# if dtype is Object, try to infer column as datetime, else numeric, ...
OBJECT_INFERENCE_HIERACHY = (
    pd.to_datetime,
    pd.to_numeric,
)

DATETIME_FMTS = (
    'datetime64[ns]',
    'datetime64[us]'
)

DICTIONARY_STRING_ABS_THRESHOLD = 30_000
DICTIONARY_STRING_PCT_THRESHOLD = 0.3


@dataclass
class ReportLine:
    """For summary of inferred schema types, NAs, nunique, etc for each column"""
  
    column: str
    dtype: str
    count: int
    na: int
    unique: int
    unique_pct: str = field(init=False)
    keyed: bool
    samples: List[Any]

    def __post_init__(self):
        self.unique_pct = f'{(100 * self.unique / (self.count - self.na)):.2f}%'


def _read_data(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, (str, Path)):
        if str(data).endswith('.csv'):
            return pd.read_csv(data)
        elif str(data).endswith('.xlsx'):
            return pd.read_excel(data)
        raise Exception(f'Unrecognized file extension {data}')
    raise Exception(f'Unrecognized type {type(data)}')
    

def _resolve_hierachy(curr_type: type, prev_type: type | None = None):
    """Get the backwards compatible type if the type of column in this 
    new dataframe doesn't match what was inferred from previous ones"""
  
    if not prev_type:
        return curr_type
    elif curr_type == prev_type:
        return curr_type
    else:
        # conflict -- resolve ancestry
        tmp = curr_type
        while (tmp := TYPE_HIERACHY.get(tmp)):
            if tmp == prev_type:
                return prev_type

        tmp = prev_type
        while (tmp := TYPE_HIERACHY.get(tmp)):
            if tmp == curr_type:
                return curr_type

        raise Exception(f'Could not resolve hierachy between {prev_type=} and {curr_type=}')


def _create_field_json(
    name: str, 
    dtype: str, 
    keyed: bool = False, 
    ordered: bool = False, 
    nullable: bool = True
) -> Dict[str, Any]:
    """Convert field values into json to write to .yaml"""
  
    res = {
        'field_name': name,
        'type': dtype,
    }
    if keyed:
        res['keyed'] = True
    if ordered:
        res['ordered'] = True
    res['nullable'] = nullable
    return res


def _to_arrow_type(s: pd.Series) -> str:
    """Infer appropriate PyArrow type based on pd.Series stats"""
  
    s = s.dropna()
    inferred_dtype = str(s.infer_objects().dtype)

    if inferred_dtype == 'object':
        for f_infer in OBJECT_INFERENCE_HIERACHY:
            try:
                s = f_infer(s)
                inferred_dtype = str(s.dtype)
                break
            except:
                pass

    if (arr_dtype := PANDAS_TO_ARROW_TYPES.get(inferred_dtype)):
        return arr_dtype
    
    elif inferred_dtype == 'object':
        if (len(s) > DICTIONARY_STRING_ABS_THRESHOLD and 
            s.nunique() < DICTIONARY_STRING_ABS_THRESHOLD) or \
           (len(s) < DICTIONARY_STRING_ABS_THRESHOLD and
            s.nunique() / len(s) < DICTIONARY_STRING_PCT_THRESHOLD) \
        :
            return 'dictionary<values=string>'
        return 'string'
        
    elif inferred_dtype in DATETIME_FMTS:
        dt_type = s.loc[s.first_valid_index()]

        if isinstance(dt_type, (datetime.datetime, pd.Timestamp)):
            return 'timestamp[us, tz=UTC]'
        elif isinstance(dt_type, datetime.date):
            return 'date32[day]' 
        raise Exception('Unexpected datetime type')

    raise Exception(f'Unexpected pandas type {inferred_dtype}')
    

class DataframeToSchema:
    """Incrementally "relax" schema as new dataframes are added"""
    
    def __init__(self):
        self.keys = set()
        self.vals = set()
        self.types = {}
        self._dfs = []


    @property
    def df(self):
        """Get the concatenated version of all added dataframes"""
      
        if not self._dfs:
            return pd.DataFrame()
        if len(self._dfs) > 1:
            self._dfs = [pd.concat(self._dfs)]
        return self._dfs[0]


    def add(
        self, 
        data: pd.DataFrame | str | Path, 
        observation_time: datetime.date | datetime.datetime | None = None
    ):
        """Edit schema by adding this new dataframe into consideration"""

        df = _read_data(data)
        if 'observation_time' not in data:
            df['observation_time'] = observation_time or datetime.datetime.now()

        if (not (_df := self.df).empty):
            df = pd.concat([
              df, _df[_df['observation_time'] == df['observation_time']]
            ])
            
        prev_vals = set(self.vals)
        prev_keys = set(self.keys)
        prev_types = copy.deepcopy(self.types)
        
        for col_name in df:
            col = df[col_name]
            n_na = col.isna().sum()
            n_dup = col.dropna().duplicated(keep=False).sum()
            
            if (
                not n_na and
                not n_dup and
                col_name not in self.vals 
                and col.dtype != 'float64'
            ):
                """
                Keys should:
                 * not contain null values
                 * not contain duplicates
                 * have no previous occurences of NAs
                 * not be continuous numeric values
                """
                self.keys.add(col_name)
            else:
                if col_name in self.keys:
                    logger.info(f'[{col_name}] has been demoted from key to value because:')
                    if n_na:
                        logger.info(f' * {n_na} NA(s) were found')
                    if n_dup:
                        logger.info(f' * {n_dup} duplicate(s) were found')
                        
                    self.keys.remove(col_name)
                self.vals.add(col_name)

            # infer type, then resolve conflicts with predecessors
            try:
                arrow_type = _to_arrow_type(col)
                self.types[col_name] = _resolve_hierachy(arrow_type, self.types.get(col_name))
            except Exception as e:
                logger.warning(e, exc_info=True)
                logger.warning('Rewinding ...')
                
                self.vals = prev_vals
                self.keys = prev_keys
                self.types = prev_types
                return

        self._dfs.append(df)
            
      
    def set_keys(self, keys: Iterable[str]) -> bool:
        """Overwrite keys if no duplicates"""
        
        dup_rows = self.df[self.df.duplicated(subset=keys, keep=False)]

        if dup_rows.empty:
            logger.info('No duplicates found. Schema is valid for the given dataframes.')
            self.vals = self.vals.union(self.keys)
            self.keys = set(keys)
            self.vals = self.vals.difference(self.keys)
            return True
        
        logger.info(
            'The following rows have duplicate keys:'
            f'{dup_rows}'
        )
        return False


    def generate_report(self, sample_size=5) -> pd.DataFrame:
        lines = []
        df = self.df
      
        for col in df:
            lines.append(ReportLine(
                column=col,
                dtype=self.types[col], 
                count=len(df),
                na=df[col].isna().sum(), 
                unique=df[col].nunique(),
                keyed=col in self.keys,
                samples=df[df[col].notnull()][col].iloc[:sample_size].to_list() or [None]*sample_size
            ))
            
        df = pd.DataFrame(lines)
        df = pd.concat([
            df.drop(columns=['samples']),
            pd.DataFrame(df['samples'].to_list(), columns=[f'sample {i}' for i in range(sample_size)])
        ], axis=1)
        
        return df.sort_values(['keyed'], ascending=False)

  
    def to_yaml(self, path: str | Path):
        fields = []

        # reference period
        fields.append(_create_field_json('reference_period', 'date32[day]', True, True, False))
        
        for col in self.keys:
            if col == 'reference_period':
                continue
            fields.append(_create_field_json(col, self.types[col], True, True, False))
            
        for col in self.vals:
            fields.append(_create_field_json(col, self.types[col]))

        # observation time
        fields.append(_create_field_json('observation_time', 'timestamp[us, tz=UTC]', False, True, False))

        table_name = str(path).split('/')[-1].split('.yaml')[0]
        with open(path, 'w') as f:
            yaml.dump({
                'table_name': table_name,
                'reference_period_field': 'reference_period', 
                'fields': fields
            }, f, default_flow_style=False, sort_keys=False)
