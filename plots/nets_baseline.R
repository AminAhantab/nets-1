library(tidyverse)

data_path <- "data/experiments/lenet_mnist_"
optimisers <- c("sgd", "adam")
targets <- c("t0.05", "t0.1", "t0.2", "t0.3")
folders <- paste(
    data_path,
    do.call(function(...) paste(..., sep = "/"), expand.grid(optimisers, targets)),
    sep = "_"
)

paste0(folders, "/nets_train.csv") |>
    map(read_csv)

adam_df <- read_csv()


nets_adam_df <- paste0(
    "data/experiments/lenet_mnist_adam/",
    targets,
    "/nets_train.csv"
) |>
    imap(~ read_csv(.x) %>% mutate(target = targets[.y])) |>
    bind_rows() |>
    mutate(optimiser = "adam", method = "nets") |>
    rename(iteration = 1)

random_adam_df <- paste0(
    "data/experiments/lenet_mnist_adam/",
    targets,
    "/rand_train.csv"
) |>
    imap(~ read_csv(.x) %>% mutate(target = targets[.y])) |>
    bind_rows() |>
    mutate(optimiser = "adam", method = "random") |>
    rename(iteration = 1)

adam_df <- bind_rows(nets_adam_df, random_adam_df)

nets_sgd_df <- paste0(
    "data/experiments/lenet_mnist_sgd/",
    targets,
    "/nets_train.csv"
) |>
    imap(~ read_csv(.x) %>% mutate(target = targets[.y])) |>
    bind_rows() |>
    mutate(optimiser = "sgd", method = "nets") |>
    rename(iteration = 1)

random_sgd_df <- paste0(
    "data/experiments/lenet_mnist_sgd/",
    targets,
    "/rand_train.csv"
) |>
    imap(~ read_csv(.x) %>% mutate(target = targets[.y])) |>
    bind_rows() |>
    mutate(optimiser = "sgd", method = "random") |>
    rename(iteration = 1)

sgd_df <- bind_rows(nets_sgd_df, random_sgd_df)

nets_combo_df <- paste0(
    "data/experiments/lenet_mnist_adam_sgd_combo/",
    targets,
    "/nets_train.csv"
) |>
    imap(~ read_csv(.x) %>% mutate(target = targets[.y])) |>
    bind_rows() |>
    mutate(optimiser = "combo", method = "nets") |>
    rename(iteration = 1)

random_combo_df <- paste0(
    "data/experiments/lenet_mnist_adam_sgd_combo/",
    targets,
    "/rand_train.csv"
) |>
    imap(~ read_csv(.x) %>% mutate(target = targets[.y])) |>
    bind_rows() |>
    mutate(optimiser = "combo", method = "random") |>
    rename(iteration = 1)

combo_df <- bind_rows(nets_combo_df, random_combo_df)

nets_df <- bind_rows(adam_df, sgd_df, combo_df) |>
    pivot_longer(
        c("train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"),
        names_to = c("dataset", "metric"),
        values_to = "value",
        names_sep = "_"
    ) |>
    pivot_wider(
        names_from = metric,
        values_from = value
    ) |>
    mutate(
        dataset = factor(dataset, levels = c("train", "val", "test")),
        optimiser = factor(optimiser, levels = c("sgd", "adam", "combo")),
        target = factor(target, levels = targets)
    )

nets_df |>
    filter(dataset == "val") |>
    filter(optimiser != "sgd") |>
    ggplot(aes(x = iteration, y = acc, colour = method)) +
    geom_line() +
    facet_grid(optimiser ~ target, scales = "free") +
    theme_bw() +
    theme(
        legend.position = "bottom",
        legend.title = element_blank()
    ) +
    labs(
        x = "Iteration",
        y = "Loss",
        colour = "Dataset",
        title = "Loss vs Iteration"
    ) +
    ylim(c(0.9, 1))
