library(tidyverse)

random_df <- read_csv("../results/nets_experiments/random_1.csv") |>
    rename(iteration = 1)
random_df

random_df %>%
    ggplot(aes(x = iteration, y = val_loss)) +
    geom_line()

random_df %>%
    ggplot(aes(x = iteration, y = test_acc)) +
    geom_line()
