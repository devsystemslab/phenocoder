def test_2d_images(example_imgs):
    imgs = example_imgs('2d')
    assert imgs.ndim == 3  # (n_images, height, width)


def test_3d_images(example_imgs):
    imgs = example_imgs('3d')
    assert imgs.ndim == 4  # (n_images, depth, height, width)


def test_2d_sdata(sample_data_2d):
    print(sample_data_2d)


def test_3d_sdata(sample_data_3d):
    print(sample_data_3d)
