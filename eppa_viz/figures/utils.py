"""Plotly trace helpers."""

import hashlib
import re

def sanitize_uid(output, region, scenario):
    """
    Create a valid CSS selector-safe UID for Plotly traces.
    Custom variables contain JSON strings with special characters that break CSS selectors.
    This function creates a sanitized version that's safe to use.
    """
    # If output is a JSON string (custom variable), hash it to create a safe identifier
    if output.startswith('{'):
        # Use a hash of the JSON string to create a unique, safe identifier
        output_hash = hashlib.md5(output.encode()).hexdigest()[:12]
        return f"custom_{output_hash}_{region}_{scenario}"
    else:
        # For regular outputs, just sanitize any problematic characters
        safe_output = re.sub(r'[^a-zA-Z0-9_-]', '_', output)
        return f"{safe_output}_{region}_{scenario}"


class TraceInfo:
    def __init__(self, figure):
        self.fig = figure
        self.traces = self.__traces()
        self.number = self.__len__()
        self.names = self.__trace_names()
        self.colors = self.__trace_colors()
        self.uid_names = self.__uid_names()
        self.custom_data = self.__custom_data()
        self.type = self.__type()

    def __getitem__(self, index):
        return self.traces[index]

    def __len__(self):
        return len(self.traces)

    def __traces(self):
        return self.fig.get("data", [])

    def __trace_colors(self):
        return [trace.get("marker", {}).get("color") for trace in self.traces]

    def __trace_names(self):
        return [trace.get("name") for trace in self.traces]

    def __uid_names(self):
        return [trace.get("uid") for trace in self.traces]

    def __custom_data(self):
        return [trace.get("customdata") for trace in self.traces]

    def __type(self):
        return [trace.get("type") for trace in self.traces]


