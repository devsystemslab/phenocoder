import shutil

from phenocoder import DatasetMerger, generator
from tests.conftest import example_3d


def test_dataset():
    pheno = example_3d()
    data_generator = generator.DatasetGenerator(
        sdata=pheno.sdata,
        sample_key='region',
        image_key='IF',
        spatial_key='spatial_index',
        table_key='nuclei_features',
        dir_output='tests/data/tmp',
    )
    data_generator.generate_dataset()
    data_generator.save_stats()
    # delete dir_output
    shutil.rmtree(data_generator.dir_output)


def test_dataset_merger():
    pheno = example_3d()
    data_generator = generator.DatasetGenerator(
        sdata=pheno.sdata,
        sample_key='region',
        image_key='IF',
        spatial_key='spatial_index',
        table_key='nuclei_features',
        dir_output='tests/data/tmp',
    )
    data_generator.generate_dataset()
    data_generator.save_stats()


if __name__ == '__main__':
    test_dataset()
