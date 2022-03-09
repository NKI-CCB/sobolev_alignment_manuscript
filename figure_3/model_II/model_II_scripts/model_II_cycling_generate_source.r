library(tidyverse)
library(dyngen)
library(anndata)
library(ggplot2)

backbone_linear()

module_info <- tribble(
    ~module_id, ~basal, ~burn, ~independence,
    "Burn1", 1, TRUE,1,
    "Burn2", 0, TRUE,1,
    "Burn3", 0, TRUE,1,
    "Burn4", 0, TRUE,1,
    "A1", 0,FALSE, 1,
    "A2", 0, FALSE, 1,
    "A3", 0, FALSE, 1,
    "A4", 0, FALSE, 1,
    "A5", 0, FALSE, 1,
    "B1",0, FALSE, 1,
    "B2", 0, FALSE, 1,
    "B3", 0, FALSE, 1,
    "B4", 0, FALSE, 1,
    "C1", 0, FALSE, 1,
    "C2", 0, FALSE, 1,
    "C3", 0, FALSE, 1,
    "C4", 0, FALSE, 1,
    "C5", 0, FALSE, 1,
    "Burn5", 1, TRUE,1,
    "Burn6", 0, TRUE,1,
    "Burn7", 0, TRUE,1,
    "Burn8", 0, TRUE,1,
    "X1", 0, FALSE,1,
    "X2", 0, FALSE,1,
    "X3", 0, FALSE,1,
    "Y1", 0, FALSE,1,
    "Y2", 0, FALSE,1,
    "Y3", 0, FALSE,1,
)

source_module_network <- tribble(
    ~from, ~to, ~effect, ~strength, ~hill,
    "Burn1", "Burn2", 1L, 1, 2,
    "Burn2", "Burn3", 1L, 1, 2,
    "Burn3", "Burn4", 1L, 1, 2,
    "Burn4", "A1", 1L, 1, 2,
    "A1", 'A2', 1L, 1, 2,
    "A2", "A3", 1L, 1, 2,
    "A3", "A4", 1L, 1, 2,
    "A4", "A5", 1L, 1, 2,
    "A5", "B1", 1L, 1, 2,
    "B1", "B2", 1L, 1, 2,
    "B2", "B3", -1L, 1, 2,
    "B3", "B4", -1L, 1, 2,
    "B4", "C1", 1L, 1, 2,
    "B1", "B3", 1L, 1, 2,
    "B2", "B4", 1L, 2, 2,
    "C1", "C2", 1L, 1, 2,
    "C2", "C3", 1L, 1, 2,
    "C3", "C4", 1L, 1, 2,
    "C4", "C5", 1L, 1, 2,
    "C5", "A1", -1L, 10, 2,
    "Burn5", "Burn6", 1L, 1, 2,
    "Burn6", "X1", 1L, 2, 2,
    "X1", 'X2', 1L, 1, 2,
    "X2", "X3", 1L, 1, 2,
    "X3", "B1", 1L, 3, 2,
    "Y1", 'Y2', 1L, 1, 2,
    "Y2", "Y3", 1L, 1, 2,
    "Y3", "C1", -1L, 3, 2,
)

source_expression_patterns <- tribble(
    ~from, ~to, ~module_progression, ~start, ~burn, ~time,
    "sBurn1", "s1", "+Burn1,+Burn2,+Burn3,+Burn4,+A1,+A2,+A3,+A4,+A5,+B1,+B2,+B3,+B4", TRUE, TRUE, 300,
    "s1", "s2", "+C1,+C2,+C3,+C4,+C5,-B1,-B2,-B3,-B4", FALSE, FALSE, 300,
    "s2", "s3", "+B1,+B2,+B3,+B4,-A1,-A2,-A3,-A4,-A5", FALSE, FALSE, 300,
    "s3", "s1", "+A1,+A2,+A3,+A4,+A5,-C1,-C2,-C3,-C4,-C5", FALSE, FALSE, 300,
    #"sBurn2", "ext1", "+Burn5,+Burn6,+Burn7,+X1,+X2,X3", TRUE, TRUE, 300,
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
pdf("../figures/model_II_source_medium_targets_5.pdf", width=12) 
plot_backbone_modulenet(source_config)
dev.off() 

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