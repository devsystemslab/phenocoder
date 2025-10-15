import spatialdata as sd
import phenocoder as phc
import numpy as np
import anndata as ad
from skimage import io
from skimage import measure
from pathlib import Path
import os
import pandas as pd

dir_images = 'tests/data/2d/imgs'
img_files = sorted(os.listdir(dir_images))
imgs = np.asarray([io.imread(Path(dir_images, file)) for file in img_files])
img_label = io.imread(Path('tests/data/2d', 'labels.tif'))
img_mask = io.imread(Path('tests/data/2d', 'mask.tif'))

df_features = pd.DataFrame(
    measure.regionprops_table(
        label_image=img_label,
        intensity_image=np.moveaxis(imgs, 0, -1),
        properties=(
            'label',
            'centroid',
            'area',
            'eccentricity',
            'intensity_mean',
            'major_axis_length',
            'minor_axis_length',
        ),
    )
).set_index('label')

features_obs = [
    'area',
    'eccentricity',
    'major_axis_length',
    'minor_axis_length',
    'centroid-0',
    'centroid-1',
]

features_X = [col for col in df_features.columns if col not in features_obs]
adata = ad.AnnData(X=df_features[features_X], obs=df_features[features_obs])

sdata = sd.SpatialData(
    images={'IF': sd.models.Image2DModel.parse(imgs, c_coords=img_files)},
    labels={
        'nuclei': sd.models.Labels2DModel.parse(img_label),
        'mask': sd.models.Labels2DModel.parse(img_mask),
    },
    tables={'nuclei_features': sd.models.TableModel.parse(adata)},
)
# Add add label shapes
sdata.shapes['nuclei_shapes'] = sd.to_polygons(sdata.labels['nuclei'])

# write sdata to tests/data/2d
sdata.write(Path('tests/data/2d', 'sdata'), overwrite=True)

# setup phenocoder
pheno = phc.Phenocoder()
pheno.add_sdata(sdata)
