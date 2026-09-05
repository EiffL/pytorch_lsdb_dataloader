"""Module-level callables so they pickle under spawn/forkserver."""


def keep_index_only(row):
    return int(row["_healpix_29"])


def id_transform(row):
    return row


def drop_odd_pixels(df, pixel):
    return df.iloc[0:0] if pixel.pixel % 2 else df


def empty_partition(df):
    return df.iloc[:0]


def arrow_threads(row):
    import pyarrow as pa

    return pa.cpu_count(), pa.io_thread_count()


class RandomTransform:
    """An augmentation with its own checkpointable random generator."""

    def __init__(self):
        import numpy as np

        self.rng = np.random.default_rng(123)

    def __call__(self, row):
        return int(row["_healpix_29"]), float(self.rng.random())

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
