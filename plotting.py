import plotly.graph_objects as go
import numpy as np
import plotly
import json
import sys
import os


plotly.io.orca.config.executable = "/home/sontung/Tools/orca"

with open('run-20200107-232324-tag-Accuracy_val.json') as json_file:
    data = json.load(json_file)
data_entry1 = [du[2] for du in data]

colors = np.random.rand()
sz = np.random.rand() * 30

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(len(data_entry1))),
        y=data_entry1,
        mode="lines",
        name="name 1"
    )
)

fig.update_layout(
    title="Accuracy on Val set",
    xaxis_title="epoch",
    yaxis_title="accuracy",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ),
    showlegend=True
)

fig.show()

if not os.path.exists("images"):
    os.mkdir("images")

fig.write_image("images/test.png")

