library(tidyverse)
library(dyngen)
library(anndata)
library(ggplot2)

module_info <- tribble(
    ~module_id, ~basal, ~burn, ~independence,
    "Ext1", 1, TRUE, 1,
    "Ext2", 0, TRUE, 1,
    "A1", 0, FALSE, 1,
    "A2", 0, FALSE, 1,
    "A3", 0, FALSE, 1,
    "A4", 0, FALSE, 1,
    "A5", 0, FALSE, 1,
    "A6", 0, FALSE, 1,
    "A7", 0, FALSE, 1,
    "A8", 0, FALSE, 1,
    "A9", 0, FALSE, 1,
    "A10", 0, FALSE, 1,
    "A11", 0, FALSE, 1,
    "A12", 0, FALSE, 1,
    "A13", 0, FALSE, 1,
    "A14", 0, FALSE, 1,
    "A15", 0, FALSE, 1,
    "Ext3", 1, TRUE, 1,
    "Ext4", 0, TRUE, 1,
    "Ext5", 1, TRUE, 1,
    "Ext6", 0, TRUE, 1,
    "Ext7", 1, TRUE, 1,
    "Ext8", 0, TRUE, 1,
    "X1", 0, FALSE, 1,
    "X2", 0, FALSE, 1,
    "X3", 0, FALSE, 1,
    "X4", 0, FALSE, 1,
    "X5", 0, FALSE, 1,
    "X6", 0, FALSE, 1,
    "X7", 0, FALSE, 1,
    "X8", 0, FALSE, 1,
    "X9", 0, FALSE, 1,
    "X10", 0, FALSE, 1,
    "X11", 0, FALSE, 1,
    "X12", 0, FALSE, 1,
    "X13", 0, FALSE, 1,
    "X14", 0, FALSE, 1,
    "X15", 0, FALSE, 1,
    "X16", 0, FALSE, 1,
    "X17", 0, FALSE, 1,
    "X18", 0, FALSE, 1,
    "X19", 0, FALSE, 1,
    "X20", 0, FALSE, 1,
    "X21", 0, FALSE, 1,
    "X22", 0, FALSE, 1,
    "X23", 0, FALSE, 1,
    "X24", 0, FALSE, 1,
    "X25", 0, FALSE, 1,
    "X26", 0, FALSE, 1,
    "X27", 0, FALSE, 1,
    "X28", 0, FALSE, 1,
    "X29", 0, FALSE, 1,
    "X30", 0, FALSE, 1,
)

source_module_network <- tribble(
    ~from, ~to, ~effect, ~strength, ~hill,
    "Ext1", "Ext2", 1L, 1, 2,
    "Ext2", "A1", 1L, 1, 2,
    "A1", 'A2', 1L, 1, 2,
    "A2", "A3", 1L, 1, 2,
    "A3", "A4", 1L, 1, 2,
    "A4", "A5", 1L, 1, 2,
    "A5", "A6", 1L, 1, 2,
    "A6", "A7", 1L, 1, 2,
    "A7", "A8", -1L, 1, 2,
    "A8", "A9", -1L, 1, 2,
    "A9", "A10", 1L, 1, 2,
    "A6", "A8", 1L, 1, 2,
    "A7", "A9", 1L, 2, 2,
    "A10", "A11", 1L, 1, 2,
    "A11", "A12", 1L, 1, 2,
    "A12", "A13", 1L, 1, 2,
    "A13", "A14", 1L, 1, 2,
    "A14", "A15", 1L, 1, 2,
    "A15", "A1", -1L, 10, 2,
    "Ext3", "Ext4", 1L, 1, 2,
    "Ext5", "Ext6", 1L, 1, 2,
    "Ext7", "Ext8", 1L, 1, 2,
    "Ext6", "X24", 1L, 8, 2,
    "X24", "X21", 1L, 3, 2,
    "X21", "X18", 1L, 6, 2,
    "X18", "X9", 1L, 3, 2,
    "X9", "X12", 1L, 6, 2,
    "Ext4", "X2", 1L, 3, 2,
    "X2", "X13", 1L, 1, 2,
    "Ext6", "X30", 1L, 1, 2,
    "Ext8", "X1", 1L, 9, 2,
    "X1", "X21", 1L, 2, 2,
    "Ext4", "X3", 1L, 6, 2,
    "X3", "X12", -1L, 1, 2,
    "X12", "X6", 1L, 10, 2,
    "X6", "X21", 1L, 5, 2,
    "X21", "X21", -1L, 3, 2,
    "X21", "X11", 1L, 10, 2,
    "X11", "X4", 1L, 9, 2,
    "X4", "X3", 1L, 8, 2,
    "X3", "X15", -1L, 9, 2,
    "Ext8", "X15", 1L, 10, 2,
    "Ext6", "X15", 1L, 5, 2,
    "X15", "X22", 1L, 4, 2,
    "X22", "X27", 1L, 1, 2,
    "X27", "X9", 1L, 2, 2,
    "X9", "X30", 1L, 2, 2,
    "X30", "X25", 1L, 5, 2,
    "Ext8", "X8", 1L, 10, 2,
    "X8", "X10", -1L, 2, 2,
    "X10", "X26", 1L, 6, 2,
    "X26", "X8", 1L, 3, 2,
    "X8", "X2", -1L, 5, 2,
    "X2", "X25", 1L, 4, 2,
    "X25", "X22", 1L, 7, 2,
    "X22", "X28", 1L, 2, 2,
    "X28", "X20", 1L, 4, 2,
    "X20", "X9", 1L, 6, 2,
    "X9", "X7", 1L, 4, 2,
    "X7", "X11", -1L, 5, 2,
    "X11", "X18", 1L, 1, 2,
    "X18", "X12", 1L, 7, 2,
    "Ext4", "X5", 1L, 1, 2,
    "X5", "X19", 1L, 9, 2,
    "Ext6", "X16", 1L, 3, 2,
    "Ext4", "X30", 1L, 10, 2,
    "X30", "X14", -1L, 3, 2,
    "X14", "X21", 1L, 6, 2,
    "X21", "X26", -1L, 2, 2,
    "X26", "X9", -1L, 9, 2,
    "Ext6", "X13", 1L, 9, 2,
    "X13", "X25", 1L, 1, 2,
    "X25", "X21", 1L, 6, 2,
)

source_expression_patterns <- tribble(
    ~from, ~to, ~module_progression, ~start, ~burn, ~time,
    "sExt", "s1", "+Ext1,+Ext2,+A1,+A2,+A3,+A4,+A5,+A6,+A7,+A8,+A9", TRUE, TRUE, 300,
    "s1", "s2", "+A10,+A11,+A12,+A13,+A14,-A6,-A7,-A8,-A9", FALSE, FALSE, 300,
    "s2", "s3", "+A6,+A7,+A8,+A9,-A1,-A2,-A3,-A4,-A5", FALSE, FALSE, 300,
    "s3", "s1", "+A1,+A2,+A3,+A4,+A5,-A10,-A11,-A12,-A13,-A14,-A15", FALSE, FALSE, 300,
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
  num_targets = 10,
  num_hks = 10,
  simulation_params = simulation_default(
    experiment_params = simulation_type_wild_type(num_simulations = 40),
    total_time = 500,
    census_interval = 1.,
  ),
  verbose = FALSE
)
pdf("../figures/model_III_source_large_targets_10.pdf", width=200, height=200) 
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

pdf("../figures/model_III_source_summary.pdf", width=200, height=200) 
plot_summary(source_model)
dev.off() 