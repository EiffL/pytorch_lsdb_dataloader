"""Module-level callables so they pickle under spawn/forkserver."""


def keep_index_only(row):
    return int(row["_healpix_29"])


def id_transform(row):
    return row


def drop_odd_pixels(df, pixel):
    return df.iloc[0:0] if pixel.pixel % 2 else df
