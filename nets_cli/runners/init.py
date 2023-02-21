import logging

from ..args import InitArgs
from .utils import hydrate_architecture, hydrate_data_dimensions
from .common import configure_logger, configure_seed, configure_torch, write_model

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

    # Initialise  model
    constructor = hydrate_architecture(architecture)
    dimensions = hydrate_data_dimensions(dataset)
    model = constructor(*dimensions, bias=False)

    # Assert for type checking
    assert isinstance(model, MaskedNetwork)

    # Initialise masks
    init_masks(model, density)

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


def init_masks(model, density: float):
    import torch
    from torch.nn import Parameter

    from nets.nn import MaskedNetwork, MaskedLayer
    from nets.utils import uniform_mask

    assert isinstance(model, MaskedNetwork)

    if density == 0.0:
        logger.info("Initialising model with all weights masked.")
        for layer in model.layers:
            assert isinstance(layer, MaskedLayer)
            mask = torch.zeros(layer.mask.shape)
            layer.mask = Parameter(mask, requires_grad=False)

        return

    if density == 1.0:
        logger.info("Initialising model with all weights unmasked.")
        for layer in model.layers:
            assert isinstance(layer, MaskedLayer)
            mask = torch.ones(layer.mask.shape)
            layer.mask = Parameter(mask, requires_grad=False)

        return

    logger.info(f"Initialising model with random masks (p = {density}).")
    for layer in model.layers:
        assert isinstance(layer, MaskedLayer)
        mask = uniform_mask(layer.mask.shape, density)
        layer.mask = Parameter(mask, requires_grad=False)
