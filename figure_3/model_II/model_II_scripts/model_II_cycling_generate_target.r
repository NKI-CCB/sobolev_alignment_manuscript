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
    "Burn5", 0, TRUE,1,
    "Burn6", 0, TRUE,1,
    "Burn7", 1, TRUE,1,
    "Burn8", 0, TRUE,1,
    "X1", 0, FALSE, 1,
    "X2", 0, FALSE, 1,
    "X3", 0, FALSE, 1,
    "Y1", 0, FALSE, 1,
    "Y2", 0, FALSE, 1,
    "Y3", 0, FALSE, 1,
)
    
target_module_network <- tribble(
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
    "X1", 'X2', 1L, 1, 2,
    "X2", "X3", 1L, 1, 2,
    "X3", "B1", 1L, 3, 2,
    "Burn7", "Burn8", 1L, 1, 2,
    "Burn8", "Y1", 1L, 2, 2,
    "Y1", 'Y2', 1L, 1, 2,
    "Y2", "Y3", 1L, 1, 2,
    "Y3", "C1", -1L, 3, 2

)

target_expression_patterns <- tribble(
    ~from, ~to, ~module_progression, ~start, ~burn, ~time,
    "sBurn1", "s1", "+Burn1,+Burn2,+Burn3,+Burn4,+A1,+A2,+A3,+A4,+A5,+B1,+B2,+B3,+B4", TRUE, TRUE, 300,
    "s1", "s2", "+C1,+C2,+C3,+C4,+C5,-B1,-B2,-B3,-B4", FALSE, FALSE, 300,
    "s2", "s3", "+B1,+B2,+B3,+B4,-A1,-A2,-A3,-A4,-A5", FALSE, FALSE, 300,
    "s3", "s1", "+A1,+A2,+A3,+A4,+A5,-C1,-C2,-C3,-C4,-C5", FALSE, FALSE, 300,
    #"sBurn2", "ext1", "+Burn5,+Burn6,+Burn7,+Y1,+Y2,Y3", TRUE, TRUE, 300,
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
# pdf("../figures/model_II/model_II_target_targets_5.pdf", width=12) 
plot_backbone_modulenet(target_config)
dev.off() 

print("###\n# START TARGET\n###")
print("TF NETWORK")
target_model <- generate_tf_network(target_config)
# pdf("../figures/model_II_target_targets_5.pdf", width=12) 
plot_backbone_modulenet(target_config)
dev.off() 

print("FEATURE NETWORK")
target_model <- generate_feature_network(target_model)
# Load feature network
# while (!file.exists("../output/model_II/source_feature_network_large.rds")){
#     Sys.sleep(.1)
# }
# target_model$feature_network <- readRDS(file="../output/model_II/source_feature_network_large.rds")
# file.remove("../output/model_II/source_feature_network_large.rds")
# Save feature network
# saveRDS(target_model$feature_network, file = "../output/target_feature_network_large_targets_5.rds")

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
# target_ad <- as_anndata(target_model)
# target_ad$write_h5ad("../output/target_dataset_large_targets_5.h5ad")
# target_ad$write_csvs("../output/target_dataset_large_targets_5.csv")
# saveRDS(target_model, file = "../output/target_model_large_targets_5.rds")
