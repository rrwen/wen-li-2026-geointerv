import geopandas as gpd
import pandas as pd

from pandas.api.types import is_numeric_dtype, is_string_dtype

# Calculates the mode for a series
def mode(x):
    out = x.map(str).value_counts().index[0]
    return out

# 25 per quantile
def quantile25(df):
    out = df.quantile(0.25)
    return out

# 50 per quantile
def quantile50(df):
    out = df.quantile(0.5)
    return out

# 75 per quantile
def quantile75(df):
    out = df.quantile(0.75)
    return out

# Calculates the counts of unique values given all possible values
def ucount(df, c, possible, round=False):
    
    # Get counts in df
    counts = df[c].value_counts().to_dict()
    
    # Round numeric values
    if round:
        possible = [int(k) if isinstance(k, Number) else k for k in possible]
        counts = {int(k) if isinstance(k, Number) else k: v for k,v in counts.items()}
    
    # Apply counts as dict with possible values
    possible = [str(k) for k in possible]
    out = {k:[0] for k in possible}
    for k, v in counts.items():
        out[str(k)] = [v]
    out = pd.DataFrame(out)
    return out

# Aggregate gdfs to a gdf of polygons representing the bins
def geobin(
    geodict,
    bins,
    stats=['sum', 'mean', 'min', 'max', 'median', 'var', 'skew', 'std', 'sem', 'mad', mode, quantile25, quantile50, quantile75],
    ucount_threshold=100,
    ignore_cols=['geometry'],
    join_kwargs={'predicate': 'intersects'},
    *args, **kwargs):
    
    # Convert to general dict if single gdf
    geodict = {'data': geodict} if isinstance(geodict, gpd.GeoDataFrame) else geodict

    # Call func if bins is not a gdf
    if not isinstance(bins, gpd.GeoDataFrame):
        bins = bins(*args, **kwargs)
        
    # Aggregate data by bins
    for name, gdf in geodict.items():
        
        # Spatially join to bins
        join = bins.sjoin(gdf, **join_kwargs)
        group = join.groupby(join.index)
        
        # Aggregate count
        counts = join.groupby(join.index).size().fillna(0)
        counts.name = f'{name}_count'
        bins = bins.join(counts)
        
        # Aggregate by stats if numeric
        num_columns = [c for c in gdf.columns if is_numeric_dtype(gdf[c]) and c not in ignore_cols]
        if len(num_columns) > 0:
            agg = group.agg({c: stats for c in num_columns})
            agg.columns = [f'{name}_{"_".join(c).strip()}' for c in agg.columns]
            agg = agg[agg.columns].apply(pd.to_numeric, errors='coerce', axis=1)
            bins = bins.join(agg)
        
        # Aggregate unique count if str and unique values under threshold
        ufreq = []
        str_columns = [c for c in gdf.columns if is_string_dtype(gdf[c]) and c not in ignore_cols]
        str_columns = [c for c in str_columns if gdf[c].unique().size <= ucount_threshold]
        if len(str_columns) > 0:
            for c in str_columns:

                # Get possible unique values in col
                possible = gdf[c].unique()

                # Count freq for each unique value
                f = group.apply(ucount, c=c, possible=possible)

                # Rename freq cols
                prefix = c.replace(f'{name}_', '')
                f.columns = [f'{name}_{c}_count' for c in f.columns]
                ufreq.append(f)

            # Combine count freq
            ufreq = pd.concat(ufreq, axis=1)
            ufreq.index = ufreq.index.get_level_values(0)
            bins = bins.join(ufreq)
        
    # Return binned aggregate data
    out = bins
    return out