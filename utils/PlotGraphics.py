import plotly.graph_objects as go
import plotly.offline
from pandas import DataFrame


class Plots:

    @staticmethod
    def plotDistributionNanValues(data: DataFrame):

        dict_nan_values = {}
        for column in data.columns:
            dict_nan_values[column] = data[column].isna().sum()
        dict_nan_values = dict(sorted(dict_nan_values.items(), key=lambda item: item[1]))

        # Plot graphic
        fig = go.Figure([go.Bar(x=list(dict_nan_values.keys()),
                                y=list(dict_nan_values.values()),
                                )])
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.6)
        fig.update_layout(plot_bgcolor="white",
                          xaxis=dict(title=dict(text="<b>Сolumn names", font=dict(size=16)), showgrid=True),
                          yaxis=dict(showgrid=True),
                          title=dict(text="The number of Nan values", font=dict(size=24)))
        fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridcolor='lightgrey')

        plotly.offline.plot(fig, filename='Nan.html', auto_open=True)

    def plotDistributedTargetVariable(self, data: DataFrame):

        pass