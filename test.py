from opsguard_ai.configuration.manager import ConfigurationManager
from opsguard_ai.components.label_alignment import LabelAlignment


config = ConfigurationManager()

label_alignment_config = (
    config.get_label_alignment_config()
)

label_alignment = LabelAlignment(
    config=label_alignment_config
)

label_alignment.initiate_label_alignment()

print("Label alignment completed.")