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

target_module_network <- tribble(
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
    "Ext8", "X18", 1L, 10, 2,
    "X18", "X4", 1L, 10, 2,
    "X4", "X27", -1L, 3, 2,
    "X27", "X7", 1L, 7, 2,
    "X7", "X22", 1L, 6, 2,
    "X22", "X14", 1L, 6, 2,
    "X14", "X23", 1L, 10, 2,
    "X23", "X8", 1L, 10, 2,
    "X8", "X30", 1L, 2, 2,
    "Ext4", "X12", 1L, 6, 2,
    "X12", "X11", 1L, 4, 2,
    "Ext4", "X7", 1L, 10, 2,
    "Ext8", "X29", 1L, 8, 2,
    "Ext6", "X18", 1L, 3, 2,
    "Ext4", "X18", 1L, 7, 2,
    "Ext6", "X28", 1L, 6, 2,
    "X28", "X19", 1L, 9, 2,
    "X19", "X26", -1L, 1, 2,
    "X26", "X26", 1L, 2, 2,
    "X26", "X18", -1L, 8, 2,
    "X18", "X21", 1L, 5, 2,
    "X21", "X28", 1L, 2, 2,
    "X28", "X2", 1L, 5, 2,
    "X2", "X17", 1L, 5, 2,
    "Ext6", "X12", 1L, 3, 2,
    "X12", "X8", -1L, 6, 2,
    "X8", "X29", 1L, 7, 2,
    "X29", "X27", 1L, 2, 2,
    "X27", "X14", -1L, 10, 2,
    "X14", "X24", 1L, 2, 2,
    "X24", "X1", 1L, 2, 2,
    "Ext4", "X6", 1L, 3, 2,
    "X6", "X4", 1L, 2, 2,
    "Ext4", "X15", 1L, 10, 2,
    "X15", "X17", 1L, 2, 2,
    "Ext8", "X3", 1L, 9, 2,
    "X3", "X30", -1L, 4, 2,
    "X30", "X6", -1L, 9, 2,
    "Ext6", "X25", 1L, 2, 2,
    "X25", "X1", 1L, 10, 2,
    "X1", "X12", 1L, 3, 2,
    "X12", "X17", 1L, 1, 2,
    "X17", "X17", 1L, 9, 2,
    "X17", "X8", 1L, 4, 2,
    "X8", "X15", 1L, 9, 2,
    "X15", "X10", 1L, 3, 2,
    "X10", "X30", -1L, 2, 2,
    "X30", "X2", 1L, 8, 2,
    "X2", "X23", -1L, 10, 2,
    "Ext6", "X27", 1L, 1, 2,
)

target_expression_patterns <- tribble(
    ~from, ~to, ~module_progression, ~start, ~burn, ~time,
    "sExt1", "s1", "+Ext1,+Ext2,+A1,+A2,+A3,+A4,+A5,+A6,+A7,+A8,+A9", TRUE, TRUE, 300,
    "s1", "s2", "+A10,+A11,+A12,+A13,+A14,-A6,-A7,-A8,-A9", FALSE, FALSE, 300,
    "s2", "s3", "+A6,+A7,+A8,+A9,-A1,-A2,-A3,-A4,-A5", FALSE, FALSE, 300,
    "s3", "s1", "+A1,+A2,+A3,+A4,+A5,-A10,-A11,-A12,-A13,-A14,-A15", FALSE, FALSE, 300,
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
  num_targets = 10, #INIT 500
  num_hks = 10, # INIT 500
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

pdf("../figures/model_III_target_targets_10.pdf", width=50, height=50) 
plot_backbone_modulenet(target_config)
dev.off() 

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

pdf("../figures/model_III_target_summary.pdf", width=50, height=50) 
plot_summary(target_model)
dev.off() 
