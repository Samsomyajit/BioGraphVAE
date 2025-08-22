from experiment_zinc import load_dataset, run_experiment

# ✅ Extract data for ZINC
loader_zinc, dim_zinc = load_dataset()
metrics_bio_zinc, x_zinc_real, x_zinc_gen, z_zinc_bio = run_experiment(loader_zinc, dim_zinc, model_name='BioGraph')
metrics_vae_zinc, x_zinc_real_v, x_zinc_gen_v, z_zinc_vae = run_experiment(loader_zinc, dim_zinc, model_name='VAE')
metrics_graph_zinc, x_zinc_real_gv, x_zinc_gen_gv, z_zinc_graph = run_experiment(loader_zinc, dim_zinc, model_name='Graph')
