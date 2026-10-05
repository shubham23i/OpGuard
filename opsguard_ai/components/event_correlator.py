from pathlib import Path
import json

import numpy as np

from opsguard_ai.entities.config_entity import EventCorrelatorConfig
from opsguard_ai.utils.exception import CustomException
from opsguard_ai.utils.logger import logging


class EventCorrelator:

    def __init__(self, config: EventCorrelatorConfig):
        self.config = config

    def create_events(self, predictions: np.ndarray):

        anomaly_indices = np.where(predictions == 1)[0]

        if len(anomaly_indices) == 0:
            return []

        events = []

        event_start = anomaly_indices[0]
        event_end = anomaly_indices[0]

        for index in anomaly_indices[1:]:

            gap = index - event_end - 1

            if gap <= self.config.max_gap:
                event_end = index

            else:
                event_length = event_end - event_start + 1

                if event_length >= self.config.min_event_length:
                    events.append({
                        "start_window": int(event_start),
                        "end_window": int(event_end),
                        "duration_windows": int(event_length)
                    })

                event_start = index
                event_end = index

        event_length = event_end - event_start + 1

        if event_length >= self.config.min_event_length:
            events.append({
                "start_window": int(event_start),
                "end_window": int(event_end),
                "duration_windows": int(event_length)
            })

        return events

    def initiate_event_correlation(self):

        try:
            input_dir = Path(self.config.input_prediction_dir)
            output_dir = Path(self.config.output_event_dir)

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            prediction_files = sorted(
                input_dir.glob("*.npy")
            )

            for prediction_file in prediction_files:

                machine_name = prediction_file.stem

                predictions = np.load(
                    prediction_file
                )

                events = self.create_events(
                    predictions
                )

                output_file = (
                    output_dir /
                    f"{machine_name}.json"
                )

                with open(
                    output_file,
                    "w",
                    encoding="utf-8"
                ) as file:

                    json.dump(
                        {
                            "machine_id": machine_name,
                            "total_anomalous_windows": int(
                                predictions.sum()
                            ),
                            "total_events": len(events),
                            "events": events
                        },
                        file,
                        indent=4
                    )

                logging.info(
                    f"Event correlation completed for "
                    f"{machine_name} | "
                    f"Anomalous windows: {predictions.sum()} | "
                    f"Events: {len(events)}"
                )

            logging.info(
                "Event correlation completed successfully."
            )

        except Exception as e:
            raise CustomException(e)