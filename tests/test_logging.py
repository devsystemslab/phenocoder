import logging

import anndata as ad
import numpy as np

import phenocoder as phc
from phenocoder.utils import quiet_spatialdata_logging

SPATIALDATA_LOGGER = 'spatialdata._logging'


def _adata():
    """Minimal AnnData with spatial coordinates, enough to build a neighbor graph."""
    rng = np.random.RandomState(0)
    adata = ad.AnnData(X=rng.rand(30, 4).astype('float64'))
    adata.obsm['spatial'] = rng.rand(30, 3) * 100
    return adata


def test_quiet_spatialdata_logging_silences_and_restores():
    """The context manager raises the log level, then restores the previous one."""
    logger = logging.getLogger(SPATIALDATA_LOGGER)
    logger.setLevel(logging.INFO)

    with quiet_spatialdata_logging():
        assert logger.level == logging.WARNING

    assert logger.level == logging.INFO


def test_quiet_spatialdata_logging_restores_on_exception():
    """The previous level is restored even if the wrapped block raises."""
    logger = logging.getLogger(SPATIALDATA_LOGGER)
    logger.setLevel(logging.INFO)

    try:
        with quiet_spatialdata_logging():
            raise RuntimeError('boom')
    except RuntimeError:
        pass

    assert logger.level == logging.INFO


def test_set_verbose_logging_reenables_info():
    """set_verbose_logging(True) makes the context manager a no-op."""
    logger = logging.getLogger(SPATIALDATA_LOGGER)
    logger.setLevel(logging.INFO)
    try:
        phc.set_verbose_logging(True)
        with quiet_spatialdata_logging():
            # verbose mode leaves the level untouched so INFO still gets through
            assert logger.level == logging.INFO
    finally:
        phc.set_verbose_logging(False)

    # back to the default: the context manager silences again
    with quiet_spatialdata_logging():
        assert logger.level == logging.WARNING


def test_graph_build_emits_no_info_by_default(caplog):
    """Building a spatial graph through phenocoder logs no INFO records."""
    from phenocoder.spatial import spatial_message_passing

    logger = logging.getLogger(SPATIALDATA_LOGGER)
    logger.setLevel(logging.INFO)
    # spatialdata's logger sets propagate=False; re-enable it so caplog can see records
    previous_propagate = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger=SPATIALDATA_LOGGER):
            spatial_message_passing(_adata(), radius=50)
        assert not [
            record for record in caplog.records if record.levelno == logging.INFO
        ]
    finally:
        logger.propagate = previous_propagate


if __name__ == '__main__':
    test_quiet_spatialdata_logging_silences_and_restores()
    test_quiet_spatialdata_logging_restores_on_exception()
    test_set_verbose_logging_reenables_info()
