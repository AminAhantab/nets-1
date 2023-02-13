library(tidyverse)
library(ggpubr)

# Load data
data <- read_csv("results/nets/results.csv") %>%
    select(-c("test_loss", "test_acc")) %>% # Remove test data (malformed)
    pivot_longer(
        cols = c("train_loss", "val_loss", "val_acc"),
        names_to = c("dataset", "metric"),
        values_to = "value",
        names_pattern = "(.+)_(.+)"
    ) %>%
    pivot_wider(
        names_from = "metric",
        values_from = "value"
    )

data %>%
    # filter(dataset == "val") %>%
    # group_by(generation) %>%
    # summarise(
    #     fitness = min(fitness)
    # ) %>%
    ggscatter(x = "generation", y = "fitness")
