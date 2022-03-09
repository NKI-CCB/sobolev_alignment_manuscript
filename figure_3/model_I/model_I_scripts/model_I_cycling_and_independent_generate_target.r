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
    
target_module_network <- tribble(
    ~from, ~to, ~effect, ~strength, ~hill,
    "Burn1", "Burn2", 1L, 1, 2,
    "Burn2", "A1", 1L, 1, 2,
    "A1", "A2", 1L, 10, 2,
    "A2", "A3", 1L, 10, 2,
    "A3", "A4", 1L, 10, 2,
    "A4", "A1", -1L, 10, 2,
    "Burn5", "Burn6", 1L, 1, 2,
    "Burn6", "Y1", 1L, 1, 2,
    "Y1", "Y2", 1L, 10, 2,
    "Y2", "Y4", 1L, 10, 2,
    "Y1", "Y3", 1L, 10, 2,
    "Y3", "Y4", -1L, 10, 2,
)

target_expression_patterns <- tribble(
  ~from, ~to, ~module_progression, ~start, ~burn, ~time,
    "sBurn", "s1", "+Burn1,+Burn2,+Burn3,+Burn4,+A1,", TRUE, TRUE, 200,
    "s1", "s2", "+A2,-A3", FALSE, FALSE, 200,
    "s2", "s3", "+A3,-A4", FALSE, FALSE, 200,
    "s3", "s1", "+A4,-A1", FALSE,FALSE,200
)

# Create target model
target_backbone <- backbone(
  module_info = module_info,
  module_network = target_module_network,
  expression_patterns = target_expression_patterns
)

target_config <- initialise_model(
  backbone = target_backbone,
  num_cells = 20000,
  num_tfs = nrow(target_backbone$module_info),
  num_targets = 5, #INIT 500
  num_hks = 5, # INIT 500
  simulation_params = simulation_default(
    experiment_params = simulation_type_wild_type(num_simulations = 40), # TO CHANGE IF NUMBER OF CELLS
    total_time = 500,
    census_interval = 1.,
  ),
  verbose = FALSE
)

print("###\n# START TARGET\n###")
print("TF NETWORK")
target_model <- generate_tf_network(target_config) 

print("FEATURE NETWORK")
target_model <- generate_feature_network(target_model)
saveRDS(target_model$feature_network, file = "../output/target_feature_network_large_targets_5.rds")

print("GENERATE KINETICS")
target_model <- generate_kinetics(target_model)

print("GOLD STANDARD")
target_model <- generate_gold_standard(target_model)

print("CELLS")
target_model <- generate_cells(target_model)

print("EXPERIMENT")
target_model <- generate_experiment(target_model)

experiment_synchronised(
    target_model, 
    realcount="regulatorycircuits_32_lung_epithelium_lung_cancer",
    map_reference_ls=TRUE
)
print("SAVE")
target_ad <- as_anndata(target_model)
target_ad$write_h5ad("../output/target_dataset_large_targets_5.h5ad")
target_ad$write_csvs("../output/target_dataset_large_targets_5.csv")
saveRDS(target_model, file = "../output/target_model_large_targets_5.rds")
