from hydra import compose, initialize
from sys import argv
initialize(config_path="config")
hprams = compose(config_name="configs", overrides=argv[1:])
