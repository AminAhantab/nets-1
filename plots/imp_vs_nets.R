library(tidyverse)

data_dir <- "data/lenet"
datasets <- c("nets_train", "nets_search", "iterative_mp", "oneshot_mp")
trials <- 1:5

# # Read all data
# do.call(function(...) paste(..., sep = "_"), expand.grid(datasets, trials)) |>
#     paste0(".csv") |>
#     map(~ file.path(data_dir, .)) |>
#     map(read_csv)

# Read evolution data
df_nets_search <- paste0("nets_search_", trials, ".csv") |>
    map(~ file.path(data_dir, .)) |>
    imap(~ read_csv(.x) |> mutate(trial = .y)) |>
    bind_rows() |>
    mutate(dataset = "nets_search")

df_nets_search |>
    group_by(trial, generation) |>
    summarise(fitness = min(fitness)) |>
    ggplot(aes(x = generation, y = fitness, color = factor(trial))) +
    geom_line()

# Read nets training data
df_nets_train <- paste0("nets_train_", trials, ".csv") |>
    map(~ file.path(data_dir, .)) |>
    imap(~ read_csv(.x) |> mutate(trial = .y)) |>
    bind_rows() |>
    mutate(dataset = "nets_train") %>%
    rename(iteration = 1)

df_nets_train |>
    filter(iteration == max(iteration)) |>
    pull(test_acc) |>
    mean()

df_nets_train |>
    filter(!is.na(test_acc)) |>
    group_by(iteration) |>
    summarise_all(mean, na.rm = TRUE) |>
    ggplot(aes(x = iteration, y = test_acc)) +
    geom_line()

# Read iterative MP data
df_iterative_mp <- paste0("iterative_mp_", trials, ".csv") |>
    map(~ file.path(data_dir, .)) |>
    imap(~ read_csv(.x) |> mutate(trial = .y)) |>
    bind_rows() |>
    mutate(dataset = "iterative_mp") |>
    rename(iteration = 1)

df_iterative_mp |>
    filter(iteration == max(iteration)) |>
    group_by(cycle, density) |>
    summarise(
        mean_test_acc = mean(test_acc),
        min_test_acc = min(test_acc),
        max_test_acc = max(test_acc),
        .groups = "drop"
    ) |>
    ggplot(aes(x = density, y = mean_test_acc)) +
    geom_line()

df_iterative_mp |>
    filter(!is.na(test_acc)) |>
    filter(cycle == 7) |>
    group_by(iteration) |>
    summarise_all(mean, na.rm = TRUE) |>
    ggplot(aes(x = iteration, y = test_acc)) +
    geom_line()

df_oneshot_mp <- paste0("oneshot_mp_", trials, ".csv") |>
    map(~ file.path(data_dir, .)) |>
    imap(~ read_csv(.x) |> mutate(trial = .y)) |>
    bind_rows() |>
    mutate(dataset = "oneshot_mp") |>
    rename(iteration = 1)

df_oneshot_mp |>
    filter(iteration == max(iteration)) |>
    filter(density == min(density)) |>
    pull(test_acc) |>
    mean()

df_oneshot_mp |>
    filter(!is.na(test_acc)) |>
    filter(density == min(density)) |>
    group_by(iteration) |>
    summarise_all(mean, na.rm = TRUE) |>
    ggplot(aes(x = iteration, y = test_acc)) +
    geom_line()

df_random <- paste0("random_", trials, ".csv") |>
    map(~ file.path(data_dir, .)) |>
    imap(~ read_csv(.x) |> mutate(trial = .y)) |>
    bind_rows() |>
    mutate(dataset = "random") |>
    rename(iteration = 1)

# Compare three
comp_nets <- read_csv("data/nets_train_1.csv") |>
    rename(iteration = 1) |>
    filter(!is.na(test_acc)) |>
    group_by(iteration) |>
    summarise(
        mean_test_acc = mean(test_acc),
        min_test_acc = min(test_acc),
        max_test_acc = max(test_acc),
        .groups = "drop"
    ) |>
    mutate(method = "nets")

comp_imp <- df_iterative_mp |>
    filter(!is.na(test_acc)) |>
    filter(cycle == 7) |>
    group_by(iteration) |>
    summarise(
        mean_test_acc = mean(test_acc),
        min_test_acc = min(test_acc),
        max_test_acc = max(test_acc),
        .groups = "drop"
    ) |>
    mutate(method = "imp")

comp_oneshot <- df_oneshot_mp |>
    filter(!is.na(test_acc)) |>
    filter(density == min(density)) |>
    group_by(iteration) |>
    summarise(
        mean_test_acc = mean(test_acc),
        min_test_acc = min(test_acc),
        max_test_acc = max(test_acc),
        .groups = "drop"
    ) |>
    mutate(method = "oneshot")

comp_random <- df_random |>
    filter(!is.na(test_acc)) |>
    group_by(iteration) |>
    summarise(
        mean_test_acc = mean(test_acc),
        min_test_acc = min(test_acc),
        max_test_acc = max(test_acc),
        .groups = "drop"
    ) |>
    mutate(method = "random")

bind_rows(comp_nets, comp_imp, comp_oneshot, comp_random) |>
    ggplot(aes(x = iteration, y = mean_test_acc, color = method)) +
    geom_line()
# geom_ribbon(aes(ymin = min_test_acc, ymax = max_test_acc), alpha = 0.2)





read_csv("data/lenet/nets_train_1.csv") |>
    rename(iteration = 1) |>
    ggplot(aes(x = iteration, y = train_loss)) +
    geom_line()
