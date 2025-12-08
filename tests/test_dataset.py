import shutil

import IPython

from phenocoder import generator
from tests.conftest import example_3d


def test_dataset():
    pheno = example_3d()
    # IPython.embed(colors='linux')
    data_generator = generator.PatchGenerator(
        dataset='test_dataset',
        sdata=pheno.sdata,
        sample_key='well',
        image_key='IF',
        spatial_key='spatial_index',
        table_key='nuclei_features',
        dir_output='tests/data/tmp',
    )
    data_generator.generate_dataset()
    shutil.rmtree(data_generator.dir_output)


if __name__ == '__main__':
    test_dataset()
