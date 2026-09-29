from pathlib import Path
import yaml
from box import ConfigBox


def read_yaml(path_to_yaml: Path) -> ConfigBox:

    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    return ConfigBox(content)


def create_directories(path_to_directories: list):

    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)