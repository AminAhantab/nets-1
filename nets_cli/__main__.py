import logging

logger = logging.getLogger("nets_cli")


if __name__ == "__main__":
    from .cli import main

    try:
        main()
    except Exception as e:
        logger.critical("An error occurred.")
        logger.critical(e)
        raise e
