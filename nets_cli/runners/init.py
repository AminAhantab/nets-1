import logging

from nets_cli.runners.methods import init

from ..args import InitArgs
from ..io import write_model

logger = logging.getLogger("nets_cli.init")


def run_init(args: InitArgs):
    # Get initialisation arguments
    dataset = args.dataset
    architecture = args.architecture
    density = args.density
    bias = args.bias

    # Initialise model
    model = init(architecture, dataset, density, bias)

    # Write model to disk
    write_model(model, args.out_path, file_name="init.pt", overwrite=True)
