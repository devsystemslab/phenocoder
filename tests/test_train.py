from phenocoder.train import train_model

if __name__ == '__main__':
    # TODO: rewrite test, no hardcoded paths
    train_model(
        dir_dataset='/pstore/data/ihb-g-deco/USERS/schulzp9/tumoroid/test_all_plates',
        conditional=True,
        n_latent_dim=128,
        n_dense_dim=256,
        n_epochs=10,
        dropout=0.5,
        beta=0.01,
        n_workers=8,
        batch_size=64,
        input_shape=(128, 128, 1),
    )
