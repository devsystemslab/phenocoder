from conftest import example_2d
from spatialdata import SpatialData


def test_2d():
    pheno = example_2d()
    assert pheno.sdata is not None
    assert isinstance(pheno.sdata, SpatialData)
    pheno.initialize_model()
    pheno.generate_dataset()
    pheno.train()
    pheno.encode()
