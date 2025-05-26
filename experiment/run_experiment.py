import json
import click
from hb.pfedhb import train


@click.command()
@click.option("--config", help="Path to the configuration file.", default=None)
def main(**kwargs):

    with open(kwargs["config"]) as f:
        kwargs = json.load(f)

    print(kwargs)
    train(kwargs)


if __name__ == "__main__":
    main()
