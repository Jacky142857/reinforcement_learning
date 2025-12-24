import pandas as pd
from collections import defaultdict
from pathlib import Path

def load_all_stock_data(stock_symbols):
    """Load all stock data from stock_data directory"""
    stock_data_dict = {}
    script_dir = Path(__file__).parent
    root_dir = script_dir
    stock_data_dir = root_dir / 'stock_data'

    for symbol in stock_symbols:
        file_path = stock_data_dir / f'{symbol}_data.csv'
        try:
            data_df = pd.read_csv(file_path)
            data_df['Date'] = pd.to_datetime(data_df['Date'], utc=True)
            data_df = data_df.sort_values('Date').reset_index(drop=True)
            stock_data_dict[symbol] = data_df
            print(f"Loaded {symbol}: {len(data_df)} rows")
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)}")

    return stock_data_dict

def check_aligned_calendar(stock_data_dict, date_col="Date", verbose=True):
    """Check if all stock data share the same ordered Date index"""
    symbols = list(stock_data_dict.keys())
    date_series = {}
    lengths = {}
    dup_dates = {}
    non_monotonic = []

    for sym in symbols:
        df = stock_data_dict[sym].copy()
        if date_col not in df.columns:
            raise ValueError(f"{sym}: missing '{date_col}' column")
        d = pd.to_datetime(df[date_col], utc=True)
        dup_mask = d.duplicated(keep=False)
        dup_dates[sym] = list(pd.to_datetime(d[dup_mask].unique()))
        if not d.is_monotonic_increasing:
            non_monotonic.append(sym)
        date_series[sym] = pd.DatetimeIndex(d)
        lengths[sym] = len(d)

    # Compute intersection and union
    common = None
    for sym in symbols:
        common = date_series[sym] if common is None else common.intersection(date_series[sym])
    union = None
    for sym in symbols:
        union = date_series[sym] if union is None else union.union(date_series[sym])

    same_ordered = True
    mismatch_symbols = []
    for sym in symbols:
        ds = date_series[sym]
        if len(ds) != len(common) or not ds.equals(common):
            same_ordered = False
            mismatch_symbols.append(sym)

    missing_dates_by_symbol = defaultdict(set)
    extra_dates_by_symbol = defaultdict(set)
    if not same_ordered:
        common_set = set(common)
        for sym in symbols:
            sset = set(date_series[sym])
            missing_dates_by_symbol[sym] = common_set - sset
            extra_dates_by_symbol[sym] = sset - common_set

    all_aligned = (len(non_monotonic) == 0 and
                   all(len(dup_dates[sym]) == 0 for sym in symbols) and
                   same_ordered)

    summary = {
        'all_aligned': all_aligned,
        'common_length': int(len(common)) if common is not None else None,
        'symbols_checked': symbols,
        'lengths': lengths,
        'dup_dates': dup_dates,
        'non_monotonic': non_monotonic,
        'date_sets_equal': same_ordered,
        'mismatch_symbols': mismatch_symbols,
        'missing_dates_by_symbol': missing_dates_by_symbol,
        'extra_dates_by_symbol': extra_dates_by_symbol,
        'common_dates': common,
        'union_dates': union
    }

    if verbose:
        print("== Calendar Alignment Check ==")
        print(f"Symbols: {len(symbols)}")
        print(f"Common calendar length: {summary['common_length']}")
        if non_monotonic:
            print(f"⚠️ Non-monotonic dates: {non_monotonic}")
        any_dups = {k: v for k, v in dup_dates.items() if len(v) > 0}
        if any_dups:
            print("⚠️ Duplicate dates found:")
            for sym, dups in any_dups.items():
                print(f"  - {sym}: {len(dups)} duplicates (first few: {dups[:3]})")
        if same_ordered:
            print("✅ All symbols share the exact same ordered Date index.")
        else:
            print("❌ Not aligned on the same ordered Date index.")
            print(f"  Mismatch symbols: {mismatch_symbols}")
            for sym in mismatch_symbols[:5]:
                miss = list(missing_dates_by_symbol[sym])
                extra = list(extra_dates_by_symbol[sym])
                print(f"  {sym}: missing {len(miss)}, extra {len(extra)}")
                if miss:
                    print(f"    e.g. missing: {sorted(miss)[:3]}")
                if extra:
                    print(f"    e.g. extra:   {sorted(extra)[:3]}")
        print("================================")

    return summary

if __name__ == "__main__":
    stock_symbols = [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO",
        "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH",
        "V", "VZ", "WBA", "WMT", "RTX"
    ]

    stock_data_dict = load_all_stock_data(stock_symbols)
    summary = check_aligned_calendar(stock_data_dict)

    if summary['all_aligned']:
        print("\n✅ All stock data are aligned on the same calendar.")
    else:
        print("\n❌ Some stock data are not aligned. See the report above.")
