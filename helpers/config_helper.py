import yaml
from pathlib import Path

class ConfigHelper:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigHelper, cls).__new__(cls)
            config_path = Path(__file__).parent.parent.resolve() / "config.yml"
            with open(config_path, "r") as f:
                cls._instance.data = yaml.safe_load(f)
        return cls._instance

    def get(self, *keys, default=None):
        """Get nested config value: get("database", "path")"""
        value = self.data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

config = ConfigHelper()
