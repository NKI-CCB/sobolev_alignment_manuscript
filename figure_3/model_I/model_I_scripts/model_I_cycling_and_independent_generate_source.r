library(tidyverse)
library(dyngen)
library(anndata)
library(ggplot2)

backbone_linear()

module_info <- tribble(
    ~module_id, ~basal, ~burn, ~independence,
    "Burn1", 1, TRUE, 1,
    "Burn2", 0, TRUE, 1,
    "Burn3", 1, TRUE, 1,
    "Burn4", 0, TRUE, 1,
    "Burn5", 1, TRUE, 1,
    "Burn6", 0, TRUE, 1,
    "A1", 0, TRUE, 1,
    "A2", 0, TRUE, 1,
    "A3", 0, TRUE, 1,
    "A4", 0, TRUE, 1,
    "X1", 0, TRUE, 1,
    "X2", 0, TRUE, 1,
    "X3", 0, TRUE, 1,
    "X4", 0, TRUE, 1,
    "Y1", 0, TRUE, 1,
    "Y2", 0, TRUE, 1,
    "Y3", 0, TRUE, 1,
    "Y4", 0, TRUE, 1
)

source_module_network <- tribble(
    ~from, ~to, ~effect, ~strength, ~hill,
    "Burn1", "Burn2", 1L, 1, 2,
    "Burn2", "A1", 1L, 1, 2,
    "A1", "A2", 1L, 10, 2,
    "A2", "A3", 1L, 10, 2,
    "A3", "A4", 1L, 10, 2,
    "A4", "A1", -1L, 10, 2,
    "Burn3", "Burn4", 1L, 1, 2,
    "Burn4", "X1", 1L, 1, 2,
    "X1", "X2", 1L, 10, 2,
    "X2", "X3", 1L, 10, 2,
    "X3", "X4", 1L, 10, 2,
    "X4", "X1", -1L, 10, 2,
)

source_expression_patterns <- tribble(
  ~from, ~to, ~module_progression, ~start, ~burn, ~time,
    "sBurn", "s1", "+Burn1,+Burn2,+Burn3,+Burn4,+A1,", TRUE, TRUE, 200,
    "s1", "s2", "+A2,-A3", FALSE, FALSE, 200,
    "s2", "s3", "+A3,-A4", FALSE, FALSE, 200,
    "s3", "s1", "+A4,-A1", FALSE,FALSE,200
)

# Create source model
source_backbone <- backbone(
  module_info = module_info,
  module_network = source_module_network,
  expression_patterns = source_expression_patterns
)

source_config <- initialise_model(
  backbone = source_backbone,
  num_cells = 20000,
  num_tfs = nrow(source_backbone$module_info),
  num_targets = 5,
  num_hks = 5,
  simulation_params = simulation_default(
    experiment_params = simulation_type_wild_type(num_simulations = 40),
    total_time = 500,
    census_interval = 1.,
  ),
  verbose = FALSE
)

print("###\n# START SOURCE\n###")
print("TF NETWORK")
source_model <- generate_tf_network(source_config)

print("FEATURE NETWORK")
source_model <- generate_feature_network(source_model)
# Save feature network
saveRDS(source_model$feature_network, file = "../output/source_feature_network_large_targets_5.rds")

print("GENERATE KINETICS")
source_model <- generate_kinetics(source_model)

print("GOLD STANDARD")
source_model <- generate_gold_standard(source_model)

print("CELLS")
source_model <- generate_cells(source_model)

print("EXPERIMENT")
source_model <- generate_experiment(source_model)
experiment_synchronised(
    source_model, 
    realcount="regulatorycircuits_32_lung_epithelium_lung_cancer",
    map_reference_ls=TRUE
)

print("SAVE")
souce_ad <- as_anndata(source_model)
souce_ad$write_h5ad("../output/source_dataset_large_targets_5.h5ad")
souce_ad$write_csvs("../output/source_dataset_large_targets_5.csv")
saveRDS(source_model, file = "../output/source_model_large_targets_5.rds")