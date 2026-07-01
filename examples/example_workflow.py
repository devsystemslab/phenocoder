"""End-to-end Phenocoder example on the bundled 3D test dataset.

This script runs the complete Phenocoder workflow on the small example dataset
shipped with the repository (``tests/data/3d/``), so it is directly runnable
without downloading any external data:

    python examples/example_workflow.py

It mirrors the narrative walkthrough in the documentation
(https://phenocoder.readthedocs.io, "Example workflow") and the integration
test in ``tests/test_workflow.py``. The steps are:

    1. Build a SpatialData object from raw images, label masks and a per-object
       feature table (see ``build_example_sdata`` below).
    2. Configure a Phenocoder instance.
    3. Generate the image-patch dataset.
    4. Initialize and train a (conditional) CVAE.
    5. Encode every object into the latent space (with spatial message passing).
    6. Cluster the latents (standard scanpy).
    7. Compute spatial neighborhood-graph statistics.
    8. Embed the per-sample statistics for comparison.

Run this from the repository root so the ``tests/data/3d/`` paths resolve.
"""

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from skimage import io

import phenocoder as phc

# Root of the bundled example dataset (relative to the repository root).
DATA_DIR = Path("tests/data/3d")


def build_example_sdata() -> sd.SpatialData:
    """Assemble a SpatialData object from the bundled 3D example files.

    The example dataset is organised per sample (a microtiter-plate *well*):

    - ``imgs/``   : one image per z-slice per channel (4 channels, per well)
    - ``labels/`` : one label mask per z-slice (integer nucleus ids)
    - ``tables/`` : one CSV per z-slice of per-object regionprops measurements

    This mirrors the ``example_3d`` fixture in ``tests/conftest.py``: the per-slice
    tables are concatenated and aggregated per object into an AnnData (features in
    ``.X``, morphometrics + keys in ``.obs``, spatial coordinates in ``.obsm``),
    and per-well image/label models are assembled into a single SpatialData object.

    Returns:
        sd.SpatialData: Images (``IF_{well}``, ``IF_MIP_{well}``), labels
            (``nuclei_{well}``) and the ``nuclei_features`` table.
    """
    # --- 1. Build the per-object table (AnnData) ---------------------------
    dir_tables = DATA_DIR / "tables"
    table_files = sorted(os.listdir(dir_tables))
    df = pd.concat([pd.read_csv(dir_tables / f) for f in table_files])

    z_step = 10
    pixel_size = 0.322
    df["centroid-0"] = df["centroid-0"] / 4
    df["centroid-1"] = df["centroid-1"] / 4
    df["z_init"] = df["z_stack"] - 1
    df["z_stack"] = df["z_stack"] / pixel_size * z_step
    df = df.groupby(["label", "well"]).mean().reset_index()

    # obs = descriptive metadata + keys; X = feature matrix
    features_obs = [
        "area",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "centroid-0",
        "centroid-1",
        "z_stack",
        "z_init",
        "well",
    ]
    features_X = [col for col in df.columns if col not in features_obs]
    adata = ad.AnnData(X=df[features_X], obs=df[features_obs])

    # Spatial coordinates in obsm
    adata.obsm["spatial"] = adata.obs[["centroid-0", "centroid-1", "z_stack"]].values
    adata.obsm["spatial_index"] = (
        adata.obs[["centroid-0", "centroid-1", "z_init"]].values.astype(int)
    )
    adata.obsm["spatial_2d"] = adata.obs[["centroid-0", "centroid-1"]].values

    # Keys SpatialData needs to link the table to the labels
    adata.obs["instance_id"] = adata.obs.index.astype(int)
    adata.obs["region"] = "nuclei"

    # --- 2. Build the per-sample image and label models -------------------
    dir_images = DATA_DIR / "imgs"
    dir_labels = DATA_DIR / "labels"
    img_files = sorted(os.listdir(dir_images))
    label_files = sorted(os.listdir(dir_labels))
    channels = [f"C0{i + 1}" for i in range(4)]

    images_dict = {}
    labels_dict = {}
    for well in adata.obs["well"].unique():
        files_well = [f for f in img_files if f"_{well}_" in f]

        # imgs: array of shape (channels, z, y, x) for this well
        imgs = []
        for channel in channels:
            files_channel = [f for f in files_well if f.endswith(f"{channel}.png")]
            imgs.append(
                np.asarray([io.imread(dir_images / f) for f in files_channel])
            )
        imgs = np.asarray(imgs)

        label_files_well = [f for f in label_files if f"_{well}_" in f]
        imgs_label = np.asarray([io.imread(dir_labels / f) for f in label_files_well])

        images_dict[f"IF_{well}"] = sd.models.Image3DModel.parse(
            imgs, c_coords=channels
        )
        # optional 2D maximum-intensity projection, handy for plotting
        images_dict[f"IF_MIP_{well}"] = sd.models.Image2DModel.parse(
            imgs.max(axis=1), c_coords=channels
        )
        labels_dict[f"nuclei_{well}"] = sd.models.Labels3DModel.parse(imgs_label)

    # --- 3. Assemble the SpatialData object -------------------------------
    sdata = sd.SpatialData(
        images=images_dict,
        labels=labels_dict,
        tables={
            "nuclei_features": sd.models.TableModel.parse(
                adata,
                region="nuclei",
                region_key="region",
                instance_key="instance_id",
                overwrite_metadata=True,
            )
        },
    )
    return sdata


def main() -> None:
    """Run the full Phenocoder pipeline on the bundled example dataset."""
    dir_dataset = "examples/output/phenocoder"

    # --- Configure Phenocoder --------------------------------------------
    sdata = build_example_sdata()
    pheno = phc.Phenocoder(
        table_key="nuclei_features",  # table in sdata.tables with per-object obs/obsm
        sample_key="well",            # obs column identifying each sample
        image_key="IF",               # images are stored as f"{image_key}_{sample}"
    )
    pheno.add_sdata(sdata)
    print(pheno)

    # --- 1. Generate the patch dataset -----------------------------------
    print("\n[1/6] Generating patch dataset ...")
    pheno.generate_dataset(
        dataset="dataset_1",
        dir_dataset=dir_dataset,
        patch_size=(32, 32),
        spatial_key_index="spatial_index",
    )

    # --- 2. Initialize a conditional CVAE --------------------------------
    print("[2/6] Initializing model ...")
    pheno.initialize_model(
        n_latent_dim=32,
        n_dense_dim=64,
        conditions=["dataset", "z"],  # pass [] for a plain (non-conditional) CVAE
        input_shape=(32, 32, 4),
    )

    # --- 3. Train ---------------------------------------------------------
    print("[3/6] Training (10 epochs for this small example) ...")
    pheno.train(n_epochs=10)

    # --- 4. Encode into latent space (with spatial message passing) ------
    print("[4/6] Encoding objects into latent space ...")
    pheno.encode(spatial_key_index="spatial_index", spatial_message_passing_radius=50)

    # --- 5. Cluster the latents, then compute spatial graph statistics ---
    print("[5/6] Clustering latents and computing spatial graph statistics ...")
    sc.pp.pca(pheno.sdata.tables["phenocoder"])
    sc.pp.neighbors(pheno.sdata.tables["phenocoder"])
    sc.tl.leiden(
        pheno.sdata.tables["phenocoder"],
        resolution=0.5,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    # This example dataset has only 3 samples (wells). To get enough
    # observations for a meaningful embedding, we compute statistics per spatial
    # *subunit* (each sample partitioned into cubes) rather than per sample. For a
    # full dataset with many samples you would typically use ``use_subunits=False``
    # and embed per sample instead (see the documentation workflow guide).
    pheno.spatialgraph_stats(
        cluster_key="leiden",
        spatial_key="spatial",
        radii=(25, 50),
        table_key="phenocoder",
        use_subunits=True,
        dim_subunit=(200, 200, 200),
        min_obs_per_subunit=10,
    )

    # --- 6. Embed the per-subunit statistics for comparison --------------
    print("[6/6] Embedding spatial-graph statistics ...")
    n_obs = pheno.adata.shape[0]
    pheno.spatialgraph_embedding(
        n_dim=min(32, n_obs - 1),
        scale=True,
        n_neighbors=min(15, n_obs - 1),  # keep n_neighbors < number of rows
        umap=True,
    )

    print("\nDone. Results:")
    print(f"  latents          -> pheno.sdata.tables['phenocoder']  "
          f"({pheno.sdata.tables['phenocoder'].shape[0]} objects)")
    print(f"  spatial embedding -> pheno.adata  ({pheno.adata.shape[0]} subunits)")


if __name__ == "__main__":
    main()
