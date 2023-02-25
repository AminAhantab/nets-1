library(tidyverse)
library(ggpubr)

# Load data
results_dir <- "../results/train_network"
architectures <- c("lenet", "conv-2", "conv-4", "conv-6")
datasets <- c("mnist", "cifar10", "cifar10", "cifar10")
results_dirs <- paste0(results_dir, "/", architectures, "/", datasets)

# Load data
df <- paste0(list.files(results_dirs, full.names = TRUE), "/results.csv") |>
    imap(\(x, idx) x |>
        read_csv() |>
        mutate(
            arch = factor(architectures[idx]),
            data = factor(datasets[idx])
        )) |>
    bind_rows() |>
    rename(iteration = 1)

df |>
    ggscatter(
        x = "iteration",
        xlab = "Training Iteration",
        y = "val_loss",
        ylab = "Validation Loss",
        color = "arch",
        add = "loess",
        facet = "data",
        alpha = 0.3,
        palette = "lancet"
    )

df |>
    ggscatter(
        x = "iteration",
        xlab = "Training Iteration",
        y = "test_acc",
        ylab = "Test Accuracy",
        color = "arch",
        facet = "data",
        panel.labs = list(data = c("MNIST", "CIFAR-10")),
        size = 1.5,
        add = "loess",
        alpha = 0.3,
        palette = "lancet"
    ) +
    scale_y_continuous(labels = scales::percent) +
    labs(color = "Architecture:")
