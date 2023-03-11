library(tidyverse)

data_path <- "data/experiments"
architecture <- c("conv_2", "lenet", "lenet", "lenet")
dataset <- c("cifar10", "mnist", "mnist", "mnist")
optimisers <- c("adam", "adam", "combo", "sgd")

targets <- c(0.1, 0.2, 0.3, 0.05)

metadata <- tibble(
    key = 1:4,
    architecture = architecture,
    dataset = dataset,
    optimiser = optimisers
)

metadata <- expand.grid(key = 1:4, target = targets) |>
    full_join(metadata) |>
    select(architecture, dataset, optimiser, target) |>
    mutate(
        path = paste0(
            data_path,
            "/",
            architecture,
            "_",
            dataset,
            "_",
            optimiser,
            "/t",
            target
        )
    )

metadata |>
    mutate(
        nets_train = paste0(path, "/nets_train.csv"),
        rand_train = paste0(path, "/rand_train.csv"),
        nets_search = paste0(path, "/nets_search.csv")
    )

train_df <- metadata |>
    mutate(
        nets_train = paste0(path, "/nets_train.csv"),
        rand_train = paste0(path, "/rand_train.csv"),
    ) |>
    select(-path) |>
    pivot_longer(
        c("nets_train", "rand_train"),
        names_to = "method",
        values_to = "path"
    ) |>
    mutate(
        df = map(path, ~ read_csv(.x) %>% rename(iteration = 1))
    ) |>
    unnest(df) |>
    select(-path) |>
    pivot_longer(
        matches("(train|test|val)_.*"),
        names_to = c("phase", "metric"),
        values_to = "value",
        names_sep = "_"
    ) |>
    pivot_wider(
        names_from = metric,
        values_from = value
    ) |>
    mutate(
        dataset = factor(dataset, levels = c("cifar10", "mnist")),
        phase = factor(phase, levels = c("train", "val", "test")),
        method = factor(case_when(
            method == "nets_train" ~ "nets",
            method == "rand_train" ~ "random"
        ), levels = c("nets", "random")),
        optimiser = factor(optimiser, levels = c("adam", "sgd", "combo")),
        architecture = factor(architecture, levels = c("conv_2", "lenet")),
        dataset = factor(dataset, levels = c("cifar10", "mnist"))
    )


train_df |>
    filter(phase == "test") |>
    filter(architecture != "lenet" | optimiser == "sgd") |>
    filter(architecture != "conv_2" | optimiser == "adam") |>
    ggplot(aes(x = iteration, y = acc, color = method)) +
    geom_line() +
    facet_grid(architecture ~ target) +
    theme_bw() +
    theme(
        legend.position = "bottom",
        legend.title = element_blank()
    )
