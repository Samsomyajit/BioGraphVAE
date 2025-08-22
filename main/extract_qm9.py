from experiment_qm9 import load_dataset, run_experiment, train_gan 

# ✅ Extract data for QM9
loader_qm9, dim_qm9 = load_dataset('QM9')
metrics_bio_qm9, x_qm9_real, x_qm9_gen, z_qm9_bio = run_experiment(loader_qm9, dim_qm9, model_name='BioGraph')
metrics_vae_qm9, x_qm9_real_v, x_qm9_gen_v, z_qm9_vae = run_experiment(loader_qm9, dim_qm9, model_name='VAE')
metrics_graph_qm9, x_qm9_real_gv, x_qm9_gen_gv, z_qm9_graph = run_experiment(loader_qm9, dim_qm9, model_name='Graph')
metrics_gan_qm9, x_qm9_real_g, x_qm9_gen_g, z_qm9_gan = train_gan(loader_qm9, dim_qm9)

