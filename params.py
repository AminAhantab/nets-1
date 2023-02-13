class HyperParameters:
    # Experiment Metadata
    name: str
    experiment: str
    seed: int
    data_dir: str
    results_dir: str

    # Data
    dataset: str
    train_size: int
    val_size: int
    test_size: int

    # Gradient Descent
    batch_size: int
    optimiser: str
    learning_rate: float

    # Neuroevolution
    pop_size: int
    elitism: int
    p_crossover: float
    mr_weight_noise: float
    mr_weight_rand: float
    mr_weight_zero: float
    parallel: bool

    # Stopping Criteria
    max_generations: int
    min_fitness: float
    min_val_loss: float

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]


def default_hyperparameters() -> HyperParameters:
    return HyperParameters(
        {
            "name": None,
            "experiment": None,
            "seed": None,
            "data_dir": "./data",
            "results_dir": "./results",
            "dataset": "mnist",
            "train_size": None,
            "val_size": None,
            "test_size": None,
            "batch_size": 64,
            "optimiser": "SGD",
            "learning_rate": 0.01,
            "pop_size": 5,
            "elitism": 2,
            "p_crossover": 0.5,
            "mr_weight_noise": 0.1,
            "mr_weight_rand": 0.1,
            "mr_weight_zero": 0.2,
            "max_generations": 20,
            "min_fitness": 0,
            "min_val_loss": 0,
            "parallel": False,
        }
    )
