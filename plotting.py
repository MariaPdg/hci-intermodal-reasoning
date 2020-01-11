import plotly.graph_objects as go
import numpy as np
import plotly
import json
import sys
import os


plotly.io.orca.config.executable = "/home/sontung/Tools/orca"
json_files = [
    ["run-20191212-131657-tag-Accuracy_val.json", "1"],
    ["run-20191218-121815-tag-Accuracy_val.json", "2"],
    ["run-20191219-111227-tag-Accuracy_val.json", "3"],
    ["run-20200104-113614-tag-Accuracy_val.json", "4"],
    ["run-20200106-234433-tag-Accuracy_val.json", "5"]
]
all_data = []
names = []
for json_file, name in json_files:
    with open("logs/%s" % json_file) as json_file:
        data = json.load(json_file)
    data_entry1 = [du[2] for du in data]
    all_data.append(data_entry1)
    names.append(name)


fig = go.Figure()

for idx, data_entry in enumerate(all_data):
    print(names[idx])
    fig.add_trace(
        go.Scatter(
            x=list(range(len(data_entry))),
            y=data_entry,
            mode="lines",
            name=names[idx]
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

