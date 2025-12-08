from spatialdata import SpatialData

from tests.conftest import example_2d


def test_2d():
    pheno = example_2d()
    assert pheno.sdata is not None
    assert isinstance(pheno.sdata, SpatialData)
    pheno.initialize_model()
    pheno.generate_dataset()
    pheno.train()
    pheno.encode()
