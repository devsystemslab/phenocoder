"""Tests for DatasetLoader.set_train_val_split batch-remainder truncation."""

import pandas as pd
import pytest

from phenocoder.generator import DatasetLoader


def _loader(counts, batch_size=64):
    """Build a DatasetLoader whose patches are pre-set to the given per-sample counts.

    set_train_val_split calls load_datasets() first, so that is stubbed out to keep the test
    focused on the truncation arithmetic rather than on file IO.
    """
    rows = []
    for sample, n in counts.items():
        rows += [
            {'well': sample, 'dataset': 'd1', 'file': f'{sample}_{i}.npy'}
            for i in range(n)
        ]
    loader = DatasetLoader(
        datasets=['d1'], dir_datasets={'d1': '/tmp/d1'}, sample_key='well'
    )
    patches = pd.DataFrame(rows)
    loader.load_datasets = lambda: setattr(loader, 'patches', patches.copy())
    return loader


def _split_sizes(loader):
    return loader.patches.groupby('split').size().to_dict()


def test_exact_multiple_of_batch_size_is_kept():
    """Regression: `iloc[:-(n % batch_size)]` is `iloc[:0]` when the remainder is 0.

    A split that was already perfectly batch-aligned -- the best case -- was silently
    emptied. With 5 samples the split is 4 train / 1 val, so 4x64 train and 1x64 val land
    on exact multiples.
    """
    loader = _loader({f'w{i}': 64 for i in range(5)})
    loader.set_train_val_split(batch_size=64)
    assert _split_sizes(loader) == {'train': 256, 'val': 64}


def test_remainder_is_dropped():
    """A non-aligned split is truncated down to a whole number of batches."""
    loader = _loader({f'w{i}': 100 for i in range(5)})
    loader.set_train_val_split(batch_size=64)
    sizes = _split_sizes(loader)
    # 4 samples x 100 = 400 train -> 384 (6 batches); 1 x 100 = 100 val -> 64 (1 batch)
    assert sizes == {'train': 384, 'val': 64}
    assert all(n % 64 == 0 for n in sizes.values())


def test_split_smaller_than_batch_size_raises():
    """Too few patches for one batch is an error, not a silently empty generator."""
    loader = _loader({f'w{i}': 10 for i in range(5)})
    with pytest.raises(ValueError, match='no complete batch'):
        loader.set_train_val_split(batch_size=64)


def test_error_names_the_starved_split_and_its_size():
    """The message should say which split is starved and how many patches it had.

    Sample-to-split assignment is shuffled, so rather than betting on which sample becomes
    the lone val sample, give every sample the same small count: whichever one is chosen,
    val gets 20 patches (no complete batch) and train gets 80 (one batch of 64, which
    survives). Only val should be reported.
    """
    loader = _loader({f'w{i}': 20 for i in range(5)})
    with pytest.raises(ValueError) as exc:
        loader.set_train_val_split(batch_size=64)
    message = str(exc.value)
    assert "'val': 20" in message
    assert 'train' not in message


def test_smaller_batch_size_rescues_a_small_dataset():
    loader = _loader({f'w{i}': 10 for i in range(5)})
    loader.set_train_val_split(batch_size=8)
    sizes = _split_sizes(loader)
    assert sizes == {'train': 40, 'val': 8}


def test_file_path_column_is_built():
    """The truncation rewrite must not disturb the file_path expansion that follows it."""
    loader = _loader({f'w{i}': 64 for i in range(5)})
    loader.set_train_val_split(batch_size=64)
    assert 'file_path' in loader.patches.columns
    assert str(loader.patches['file_path'].iloc[0]).startswith('/tmp/d1/')
