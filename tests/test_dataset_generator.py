import sys
import os
import shutil
sys.path.append('/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen')
from whole_mount_tumoroid.phenocoder.generator import DatasetGenerator, DatasetMerger

def test_dataset_generator_and_merging(delete_test_dir=True):
    dir_test = '/pstore/data/ihb-g-deco/USERS/schulzp9/tumoroid/test_qc'
    dir_screen = '/pstore/data/ihb-tumoroid/data/processed/timecourse'
    if os.path.exists(dir_test):
        shutil.rmtree(dir_test)
    plates = ['001','002', '003','004', '005']

    for plate in plates:
        dataset_generator = DatasetGenerator(dir_input=f'{dir_screen}/{plate}/{plate}-01/TIF_OVR_BG',
                                             dir_output=f'{dir_test}/{plate}',
                                             max_workers=12,
                                             dir_segmented=f'{dir_screen}/{plate}/{plate}-01/features/nuclei/TIF_OVR_BG',
                                             mode='segmented',
                                             channels = ["01"])
        dataset_generator.generate_dataset(sampling_frac=None, per_channel=True, qc_path='/pstore/data/ihb-g-deco/USERS/schulzp9/tumoroid/qc_test.csv') #, n_patches=500)
        dataset_generator.save_stats()


    dataset_merger = DatasetMerger(datasets=plates,dir_datasets=dir_test)
    dataset_merger.merge_datasets()
    # delete the test directory
    if delete_test_dir:
        shutil.rmtree(dir_test)



if __name__ == '__main__':
    test_dataset_generator_and_merging(delete_test_dir=False)


