import numpy as np
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio


def plot_test_markup(data, y_true, y_pred,
                     depth_name,
                     variable_for_markup,
                     out_dir):

    # name_columns = []

    rows, cols = 1, 2

    colors = ["peru", "grey", "olive", "orange", "lightblue"]
    color_discrete_map = {k: v for k, v in zip(range(len(np.unique(y_true))), colors)}

    Fig = make_subplots(rows=rows, cols=cols, # subplot_titles=[k for k in ['Разметка', method_name]],
                        horizontal_spacing=0.015, shared_yaxes=True)

    fig_1 = px.area(data, x=data[variable_for_markup], y=(-1) * data[depth_name], color=y_true, # data['goal'],
                    orientation='h', width=700, height=900,
                    color_discrete_map={0: 'peru', 1: 'grey'})
    for j in range(len(color_discrete_map)):
        Fig.add_trace(fig_1['data'][j], 1, 1)

    fig_2 = px.area(data, x=data[variable_for_markup], y=(-1) * data[depth_name], color=y_pred.round().ravel(),
                    orientation='h', width=700, height=900,
                    color_discrete_map={0: 'peru', 1: 'grey'})
    for j in range(len(color_discrete_map)):
        Fig.add_trace(fig_2['data'][j], 1, 2)

    for i in range(1, cols + 1):
        Fig.update_xaxes(range=[0, 1], showgrid=True, row=1, col=i, color='black', nticks=10, gridcolor='lightgrey')
        Fig.update_yaxes(showgrid=True, row=1, col=i, color='black', nticks=50, gridcolor='lightgrey')
        Fig.update_layout(legend_orientation='h', width=1100, height=1400, margin=dict(l=0, r=0, t=50, b=0))

    Fig.update_layout(plot_bgcolor='white',
                      showlegend=False)

    pio.write_image(Fig, Path(out_dir, 'Markup.jpg'), scale=6, width=800, height=1100)