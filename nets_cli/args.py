from argparse import ArgumentParser
from dataclasses import dataclass
import os


SUBCOMMANDS = ["init", "train", "search", "prune"]
OPTIMISERS = ["sgd", "adam"]
CRITERIA = ["magnitude", "random"]
DATASETS = ["mnist", "cifar10"]
ARCHITECTURES = ["lenet", "conv-2", "conv-4", "conv-6", "resnet-18", "vgg-19"]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    # add subcommands
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="valid subcommands",
        help="additional help",
        dest="subcommand",
    )

    # init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="initialise a random model",
    )

    init_parser = add_init_args(init_parser)

    # search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="search for an initialisation",
    )

    # add search arguments
    search_parser = add_search_args(search_parser)

    # train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="train a model",
    )

    # add train arguments
    train_parser = add_train_args(train_parser)

    # prune subcommand
    prune_parser = subparsers.add_parser(
        "prune",
        help="prune a model",
    )

    # add prune arguments
    prune_parser = add_prune_args(prune_parser)

    # imp subcommand
    imp_parser = subparsers.add_parser(
        "imp",
        help="iterative magnitude pruning",
    )

    # add imp arguments
    imp_parser = add_iterative_magnitude_pruning_args(imp_parser)

    return parser


@dataclass
class BaseArgs:
    """Base arguments for all subcommands."""

    subcommand: str
    """Subcommand to run."""

    seed: int
    """Random seed to use."""

    log_level: str
    """Logging level to use."""
    log_format: str
    """Logging format to use."""
    log_file: str
    """File to log to."""

    out_path: str
    """Path to save output to."""

    def __post_init__(self):
        assert self.subcommand in SUBCOMMANDS

        errors = []
        if self.log_level not in LOG_LEVELS:
            errors.append(f"invalid log level: {self.log_level}")

        if self.log_file is not None:
            if not os.path.exists(os.path.dirname(self.log_file)):
                errors.append(f"log file directory does not exist: {self.log_file}")

        if self.seed is not None and self.seed < 0:
            errors.append(f"invalid seed: {self.seed}")

        # out path should either not exist or be a directory
        if os.path.exists(self.out_path):
            if not os.path.isdir(self.out_path):
                errors.append(f"output path is not a directory: {self.out_path}")

        if len(errors) > 0:
            raise ValueError("\n".join(errors))


def add_common_args(parser):
    # out_path
    parser.add_argument(
        "--out_path",
        type=str,
        default=".",
        help="path to save output to",
        metavar="PATH",
    )

    # seed (int)
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        metavar="SEED",
    )

    # log_level (enum: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="log level",
        choices=LOG_LEVELS,
    )

    # log_format
    parser.add_argument(
        "--log_format",
        type=str,
        default="[%(levelname)s] %(message)s",
        help="log format",
    )

    # log_file
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="log file",
        metavar="FILE",
    )

    return parser


@dataclass
class InitArgs(BaseArgs):
    """Arguments for initialising a model."""

    dataset: str
    """Dataset to train on."""
    architecture: str
    """Architecture to use for the model."""
    density: float
    """Density of the model."""
    bias: bool
    """Whether to use bias in the model."""

    def __post_init__(self):
        errors = []
        if self.architecture not in ARCHITECTURES:
            errors.append(f"invalid architecture: {self.architecture}")

        if self.dataset not in DATASETS:
            errors.append(f"invalid dataset: {self.dataset}")

        if self.density < 0 or self.density > 1:
            errors.append(f"density must be in [0, 1]: {self.density}")

        if len(errors) > 0:
            raise ValueError("\n".join(errors))


def add_init_args(parser: ArgumentParser):
    # data (enum: mnist, cifar10, cifar100, imagenet)
    parser.add_argument(
        "--data",
        dest="dataset",
        type=str,
        default="mnist",
        help="dataset to train on",
        choices=DATASETS,
    )

    # architecture (enum: lenet, resnet18, resnet50, resnet101, resnet152)
    parser.add_argument(
        "--arch",
        dest="architecture",
        type=str,
        default="lenet",
        help="model to train",
        choices=ARCHITECTURES,
    )

    # density (float)
    parser.add_argument(
        "--density",
        type=float,
        default=1,
        help="density of the initial network",
        metavar="D",
    )

    # bias (bool)
    parser.add_argument(
        "--bias",
        action="store_true",
        help="whether to use bias in the model",
        default=False,
    )

    # Add common args
    parser = add_common_args(parser)

    return parser


@dataclass
class GradientDescentArgs(BaseArgs):
    """Arguments for training a model."""

    dataset: str
    """Dataset to train on."""

    optimiser: str
    """Optimiser to use for training."""
    learning_rate: float
    """Learning rate for training."""
    max_epochs: int
    """Number of epochs to train for."""
    max_iterations: int
    """Maximum number of iterations to train for."""
    batch_size: int
    """Batch size for training."""
    no_cuda: bool
    """Disable CUDA."""
    val_size: int
    """Validation set size."""

    log_every: int
    """Log every n batches."""
    log_val_every: int
    """Log validation metrics every n batches."""
    log_test_every: int
    """Log test metrics every n batches."""

    def __post_init__(self):
        errors = []
        if self.dataset not in DATASETS:
            errors.append(f"invalid dataset: {self.dataset}")

        if self.optimiser not in OPTIMISERS:
            errors.append(f"invalid optimiser: {self.optimiser}")

        if self.learning_rate <= 0:
            errors.append(f"invalid learning rate: {self.learning_rate}")

        if self.max_iterations is not None and self.max_iterations <= 0:
            errors.append(f"invalid number of iterations: {self.max_iterations}")

        if self.max_epochs is not None and self.max_epochs <= 0:
            errors.append(f"invalid number of epochs: {self.max_epochs}")

        # exactly one of max_iterations and max_epochs should be set
        if self.max_iterations is None and self.max_epochs is None:
            errors.append("either max_iterations or max_epochs must be set")
        elif self.max_iterations is not None and self.max_epochs is not None:
            errors.append("only one of max_iterations or max_epochs can be set")

        if self.batch_size <= 0:
            errors.append(f"invalid batch size: {self.batch_size}")

        if self.val_size < 0:
            errors.append(f"invalid validation set size: {self.val_size}")

        if self.log_every is not None and self.log_every < 0:
            errors.append(f"invalid log_every: {self.log_every}")

        if self.log_val_every is not None and self.log_val_every < 0:
            errors.append(f"invalid log_val_every: {self.log_val_every}")

        if self.log_test_every is not None and self.log_test_every < 0:
            errors.append(f"invalid log_test_every: {self.log_test_every}")

        if len(errors) > 0:
            raise ValueError("\n".join(errors))


def add_gradient_descent_args(parser: ArgumentParser):
    # data (enum: mnist, cifar10, cifar100, imagenet)
    parser.add_argument(
        "--data",
        dest="dataset",
        type=str,
        default="mnist",
        help="dataset to train on",
        choices=DATASETS,
    )

    # optimiser (enum: sgd, adam)
    parser.add_argument(
        "--opt",
        dest="optimiser",
        type=str,
        default="sgd",
        help="optimiser to use for training",
        choices=OPTIMISERS,
    )

    # learning_rate (float)
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=0.001,
        help="learning rate for training",
        metavar="LR",
    )

    # epochs and iterations are mutually exclusive
    group = parser.add_mutually_exclusive_group()

    # epochs (int)
    group.add_argument(
        "--epochs",
        dest="max_epochs",
        type=int,
        help="number of epochs to train for",
    )

    # iterations (int)
    group.add_argument(
        "--iterations",
        dest="max_iterations",
        type=int,
        help="maximum number of iterations to train for",
    )

    # batch_size (int)
    parser.add_argument(
        "--batch",
        dest="batch_size",
        type=int,
        default=128,
        help="batch size for training",
    )

    # no_cuda
    parser.add_argument(
        "--no_cuda",
        dest="no_cuda",
        action="store_true",
        help="disable CUDA",
    )

    # val_size
    parser.add_argument(
        "--val_size",
        dest="val_size",
        type=int,
        default=5_000,
        help="validation set size",
        metavar="COUNT",
    )

    # log_every
    parser.add_argument(
        "--log_every",
        dest="log_every",
        type=int,
        default=None,
        help="log every n batches",
        metavar="N",
    )

    # log_val_every
    parser.add_argument(
        "--log_val_every",
        dest="log_val_every",
        type=int,
        default=None,
        help="log validation every n batches",
        metavar="N",
    )

    # log_val_every
    parser.add_argument(
        "--log_test_every",
        dest="log_test_every",
        type=int,
        default=None,
        help="log validation every n batches",
        metavar="N",
    )

    # Add common args
    parser = add_common_args(parser)

    return parser


@dataclass
class TrainArgs(GradientDescentArgs):
    """Arguments for training a model."""

    model_path: str
    """Path to model to train."""

    csv_path: str
    """Path to csv file to save results to."""

    def __post_init__(self):
        super().__post_init__()

        errors = []
        if not os.path.isfile(self.model_path):
            errors.append(f"model path does not exist: {self.model_path}")

        if len(errors) > 0:
            raise ValueError("\n".join(errors))


def add_train_args(parser: ArgumentParser):
    # model_path
    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="path to model to train",
        metavar="FILE",
    )

    # Add gradient descent args
    parser = add_gradient_descent_args(parser)

    # csv path
    parser.add_argument(
        "--csv_path",
        dest="csv_path",
        type=str,
        default="",
        help="path to csv file to save results to",
        metavar="FILE",
    )

    return parser


@dataclass
class SearchArgs(GradientDescentArgs):
    """Arguments for searching for an initialisation."""

    arch: str
    """Architecture to search for."""
    initial_density: float
    """Initial density of connections."""
    target_density: float
    """Target density of connections."""

    population_size: int
    """Number of individuals in the population."""
    max_generations: int
    """Number of generations to evolve for."""

    mr_disable: float
    """Probability of disabling a connection."""
    mr_random: float
    """Probability of mutating a connection to a random value."""
    mr_noise: float
    """Probability of mutating a connection by adding noise."""
    mr_noise_scale: float
    """Scale of noise to add to a connection."""

    p_crossover: float
    """Probability of fittest parent genes transferred during crossover."""
    num_elites: int
    """Number of fittest individuals to keep in the population."""

    min_fitness: float
    """Minimum fitness to stop searching at."""

    csv_path: str
    """Path to csv file to save results to."""

    def __post_init__(self):
        errors = []
        if self.arch not in ARCHITECTURES:
            errors.append(f"invalid architecture: {self.arch}")

        if self.initial_density <= 0 or self.initial_density > 1:
            errors.append(f"invalid initial density: {self.initial_density}")

        if self.target_density <= 0 or self.target_density > 1:
            errors.append(f"invalid target density: {self.target_density}")

        if self.population_size <= 0:
            errors.append(f"invalid population size: {self.population_size}")

        if self.max_generations is not None and self.max_generations <= 0:
            errors.append(f"invalid number of generations: {self.max_generations}")

        if self.mr_disable < 0 or self.mr_disable > 1:
            errors.append(f"invalid disable mutation rate: {self.mr_disable}")

        if self.mr_random < 0 or self.mr_random > 1:
            errors.append(f"invalid random mutation rate: {self.mr_random}")

        if self.mr_noise < 0 or self.mr_noise > 1:
            errors.append(f"invalid noise mutation rate: {self.mr_noise}")

        if self.mr_noise_scale <= 0:
            errors.append(f"invalid noise scale: {self.mr_noise_scale}")

        if self.p_crossover < 0 or self.p_crossover > 1:
            errors.append(f"invalid crossover probability: {self.p_crossover}")

        if self.num_elites < 0:
            errors.append(f"invalid number of elites: {self.num_elites}")

        if len(errors) > 0:
            raise ValueError("\n".join(errors))


def add_search_args(parser: ArgumentParser):
    # arch (str)
    parser.add_argument(
        "--arch",
        dest="arch",
        type=str,
        required=True,
        help="architecture to search for",
        metavar="ARCH",
    )

    # initial_density (float)
    parser.add_argument(
        "--init_density",
        dest="initial_density",
        type=float,
        default=0.5,
        help="initial density of connections",
        metavar="DENSITY",
    )

    # target_density (float)
    parser.add_argument(
        "--target_density",
        dest="target_density",
        type=float,
        default=0.2,
        help="target density of connections",
        metavar="DENSITY",
    )

    # population_size
    parser.add_argument(
        "--pop",
        dest="population_size",
        type=int,
        default=100,
        help="population size for search",
        metavar="POP",
    )

    # generations
    parser.add_argument(
        "--gens",
        dest="max_generations",
        type=int,
        help="number of generations to search for",
        metavar="GEN",
    )

    # mr_disable (float)
    parser.add_argument(
        "--mr_disable",
        type=float,
        default=0.1,
        help="probability of disabling a gene",
        metavar="DISABLE",
    )

    # mr_random (float) mutation rate
    parser.add_argument(
        "--mr_random",
        type=float,
        default=0.1,
        help="probability of mutating a gene with a new random value",
        metavar="RAND",
    )

    # mr_noise (float) mutation rate
    parser.add_argument(
        "--mr_noise",
        type=float,
        default=0.1,
        help="probability of mutating a gene with a small noise value",
        metavar="NOISE",
    )

    # mr_noise_scale (float) mutation rate
    parser.add_argument(
        "--mr_noise_scale",
        type=float,
        default=0.1,
        help="scale of noise to add to a gene when mutating",
        metavar="MAG",
    )

    # p_crossover (float)
    parser.add_argument(
        "--p_crossover",
        type=float,
        default=0.5,
        help="probability genes from fitter parent are passed to offspring",
        metavar="PROB",
    )

    # num_elites (int)
    parser.add_argument(
        "--elites",
        dest="num_elites",
        type=int,
        default=1,
        help="number of fittest individuals to keep in the population",
        metavar="ELITES",
    )

    # min_fitness (float)
    parser.add_argument(
        "--min_fitness",
        type=float,
        help="minimum fitness to stop searching at",
        metavar="FITNESS",
    )

    # csv_path (str)
    parser.add_argument(
        "--csv_path",
        dest="csv_path",
        type=str,
        help="path to csv file to save results to",
        metavar="CSV",
    )

    # Add training args
    parser = add_gradient_descent_args(parser)

    return parser


@dataclass
class PruneArgs(BaseArgs):
    """Arguments for pruning a model."""

    model_path: str
    """Path to model to prune."""
    dataset: str
    """Dataset to evaluate pruning on."""

    criterion: str
    """Pruning criterion."""

    threshold: float
    """Pruning threshold."""
    count: int
    """Number of connections to prune."""
    fraction: float
    """Fraction of connections to prune."""

    def __post_init__(self):
        errors = []
        if not os.path.isfile(self.model_path):
            errors.append(f"invalid model path: {self.model_path}")

        if self.dataset is not None and self.dataset not in DATASETS:
            errors.append(f"invalid dataset: {self.dataset}")

        if self.criterion not in CRITERIA:
            errors.append(f"invalid pruning criterion: {self.criterion}")

        if self.threshold is not None and self.threshold <= 0:
            errors.append(f"invalid pruning threshold: {self.threshold}")

        if self.count is not None and self.count <= 0:
            errors.append(f"invalid number of connections to prune: {self.count}")

        if self.fraction is not None and self.fraction <= 0:
            errors.append(f"invalid fraction of connections to prune: {self.fraction}")

        one_of = [self.threshold, self.count, self.fraction]
        if len([x for x in one_of if x is not None]) != 1:
            errors.append(f"must specify exactly one of threshold, count or fraction")

        if len(errors) > 0:
            raise ValueError("\n".join(errors))


def add_prune_args(parser: ArgumentParser):
    # model_path (str)
    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="path to model to prune",
        metavar="PATH",
    )

    # data (enum: mnist, cifar10, cifar100) (optional)
    parser.add_argument(
        "--data",
        dest="dataset",
        type=str,
        help="dataset to evaluate pruning on",
        choices=DATASETS,
    )

    # criterion (enum: magnitude, l1)
    parser.add_argument(
        "--criterion",
        type=str,
        default="magnitude",
        help="pruning criterion",
        choices=CRITERIA,
    )

    group = parser.add_mutually_exclusive_group()

    # threshold (float)
    group.add_argument(
        "--threshold",
        type=float,
        help="pruning threshold",
        metavar="T",
    )

    # prune (int)
    group.add_argument(
        "--count",
        type=int,
        help="number of connections to prune",
        metavar="N",
    )

    # fraction (float)
    group.add_argument(
        "--fraction",
        type=float,
        help="fraction of connections to prune",
        metavar="PERC",
    )

    # Add common args
    parser = add_common_args(parser)

    return parser


@dataclass
class IterativeMagnitudePruningArgs(TrainArgs):
    """Arguments for iterative magnitude pruning."""

    criterion: str
    """Pruning criterion."""

    threshold: float
    """Pruning threshold."""
    count: int
    """Number of connections to prune."""
    fraction: float
    """Fraction of connections to prune."""

    cycles: int
    """Number of pruning cycles to perform."""

    reinit: bool
    """Whether to reinitialize weights after pruning."""

    def __post_init__(self):
        super().__post_init__()
        errors = []

        if self.criterion not in CRITERIA:
            errors.append(f"invalid pruning criterion: {self.criterion}")

        if self.threshold is not None and self.threshold <= 0:
            errors.append(f"invalid pruning threshold: {self.threshold}")

        if self.count is not None and self.count <= 0:
            errors.append(f"invalid number of connections to prune: {self.count}")

        if self.fraction is not None and self.fraction <= 0:
            errors.append(f"invalid fraction of connections to prune: {self.fraction}")

        one_of = [self.threshold, self.count, self.fraction]
        if len([x for x in one_of if x is not None]) != 1:
            errors.append(f"must specify exactly one of threshold, count or fraction")

        if self.cycles <= 0:
            errors.append(f"invalid number of pruning cycles: {self.cycles}")

        if len(errors) > 0:
            raise ValueError("\n".join(errors))


def add_iterative_magnitude_pruning_args(parser: ArgumentParser):
    # Add training args
    parser = add_train_args(parser)

    # Add pruning args
    # criterion (enum: magnitude, l1)
    parser.add_argument(
        "--criterion",
        type=str,
        default="magnitude",
        help="pruning criterion",
        choices=CRITERIA,
    )

    group = parser.add_mutually_exclusive_group()

    # threshold (float)
    group.add_argument(
        "--threshold",
        type=float,
        help="pruning threshold",
        metavar="T",
    )

    # prune (int)
    group.add_argument(
        "--count",
        type=int,
        help="number of connections to prune",
        metavar="N",
    )

    # fraction (float)
    group.add_argument(
        "--fraction",
        type=float,
        help="fraction of connections to prune",
        metavar="PERC",
    )

    # cycles (int)
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="number of pruning cycles to perform",
        metavar="CYCLES",
    )

    # reinit (bool)
    parser.add_argument(
        "--reinit",
        action="store_true",
        default=True,
        help="whether to reinitialize weights after pruning",
    )

    return parser
