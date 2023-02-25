import logging

from ..args import InitArgs
from ..config import configure_logger, configure_seed, configure_torch
from ..io import write_model
from ..hydrator import hydrate_new_model

logger = logging.getLogger("nets_cli.init")


def run_init(args: InitArgs):
    # Configure environment
    configure_logger(args)
    configure_torch()
    configure_seed(args.seed)

    from nets.nn import MaskedNetwork

    # Get relevant arguments
    dataset = args.dataset
    architecture = args.architecture
    density = args.density

    # Initialise model
    model = hydrate_new_model(architecture, dataset, density, False)
    assert isinstance(model, MaskedNetwork)

    # Log model creation
    logger.info(
        "Initialised a %s (density %.0f%%) for learning %s classifications.",
        architecture,
        model.density() * 100,
        dataset,
    )
    logger.debug("Initialised model: %s", model)

    # Write model to disk
    write_model(model, args.out_path, file_name="init.pt", overwrite=True)
