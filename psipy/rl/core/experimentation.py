from typing import Any

import yaml


class SerializableMixin:
    def serialize(self, file_path):
        # Collect all attributes in a dictionary
        attributes = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

        # Write the attributes to a YAML file
        with open(file_path, "w") as file:
            yaml.dump(attributes, file)

    def deserialize(self, file_path):
        # Read the attributes from a YAML file
        with open(file_path, "r") as file:
            attributes = yaml.safe_load(file)

        # Set the attributes to the instance
        for attr, value in attributes.items():
            setattr(self, attr, value)


class Environment(SerializableMixin):
    def __init__(self):
        self.attrs: dict[str, Any] = {}
        pass


class Experiment(SerializableMixin):
    def __init__(self):
        self.attrs: dict[str, Any] = {}
        pass
