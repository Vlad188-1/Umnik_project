import plotly.graph_objects as go
import plotly.offline
from pandas import DataFrame, Series
import numpy as np
from plotly.subplots import make_subplots
from pathlib import Path


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

        plotly.offline.plot(fig, filename='plots/Nan.html', auto_open=True)

    @staticmethod
    def plotDistributedTargetVariable(y):
        from collections import Counter
        from numpy import ndarray
        if isinstance(y, Series):
            count_values = y.value_counts().to_dict()
        elif isinstance(y, ndarray):
            count_values = Counter(y)
        name_classes, count_classes = count_values.keys(), count_values.values()

        fig = go.Figure([go.Bar(x=list(name_classes),
                                y=list(count_classes),
                                )])
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.6)
        fig.update_layout(plot_bgcolor="white",
                          xaxis=dict(title=dict(text="<b>Name classes", font=dict(size=16)), showgrid=True),
                          yaxis=dict(showgrid=True),
                          title=dict(text="Distribution target variable", font=dict(size=24)))
        fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridcolor='lightgrey')
        plotly.offline.plot(fig, filename="plots/TargetVariable.html", auto_open=True)


def plotTrainValidCurve(history, writer):
    # print('Plotting training and validation curve...')

    acc = history['train_accuracy']
    val_acc = history['val_accuracy']
    loss = history['train_loss']
    val_loss = history['val_loss']

    epochs = np.arange(1, len(acc) + 1)

    # Plot loss and accuracy curves
    fig = make_subplots(rows=1, cols=2, subplot_titles=['$\Large\\textbf{Accuracy}$', '$\Large\\textbf{Loss}$'],
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=epochs, y=acc, mode='lines', showlegend=True, name='Training',
                             marker_color='blue'),1,1)
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', showlegend=True, name='Validation',
                             marker_color='red'),1,1)

    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', showlegend=False, marker_color='blue'),1,2)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', showlegend=False, marker_color='red'),1,2)

    # fig.add_trace(go.Scatter(x=[epochs[val_loss.index(min(val_loss))]],
    #                          y=[val_acc[epochs[val_loss.index(min(val_loss)) - 1]]], marker_color='darkgreen',
    #                          name='Accuracy для<br> минимального loss'), 1, 1)
    #
    # fig.add_trace(go.Scatter(x=[epochs[val_loss.index(min(val_loss))]],
    #                          y=[val_loss[epochs[val_loss.index(min(val_loss)) - 1]]], marker_color='black',
    #                          name='Минимальный loss'), 1, 2)
    #
    # fig.add_shape(type="line", row=1, col=2,
    #               x0=epochs[val_loss.index(min(val_loss))], y0=0.3, x1=epochs[val_loss.index(min(val_loss))], y1=0.1,
    #               line=dict(color="black", width=2, dash="dashdot"))
    #
    # fig.add_shape(type="line", row=1, col=1,
    #               x0=epochs[val_loss.index(min(val_loss))], y0=0.96, x1=epochs[val_loss.index(min(val_loss))], y1=0.86,
    #               line=dict(color="black", width=2, dash="dashdot"))

    fig.update_layout(plot_bgcolor='white', title=dict(text='<b>Обучение и валидация нейронной сети', x=0.5,
                                                       font=dict(size=20)),
                      legend=dict(font=dict(size=14)), height=550, width=1150, showlegend=True)
    fig.update_xaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey',
                     title=dict(text='Epochs', font=dict(size=23)))
    fig.update_xaxes(row=1, col=2, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey',
                     title=dict(text='Epochs', font=dict(size=23)))

    fig.update_yaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')
    fig.update_yaxes(row=1, col=2, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')

    fig.write_image(Path(writer.log_dir, "train_val_curve.jpg"))


def plotTrainValidCurveAE(history, writer):

    loss = history['train_loss']
    val_loss = history['val_loss']

    epochs = np.arange(1, len(loss) + 1)

    # Plot loss and accuracy curves
    fig = make_subplots(rows=1, cols=1, subplot_titles=['$\Large\\textbf{Accuracy}$', '$\Large\\textbf{Loss}$'],
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', showlegend=False, marker_color='blue'), 1, 1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', showlegend=False, marker_color='red'), 1, 1)

    fig.update_layout(plot_bgcolor='white', title=dict(text='<b>Обучение и валидация нейронной сети', x=0.5,
                                                       font=dict(size=20)),
                      legend=dict(font=dict(size=14)), height=550, width=1150, showlegend=True)
    fig.update_xaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey',
                     title=dict(text='Epochs', font=dict(size=23)))

    fig.update_yaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')

    fig.write_image(Path(writer.log_dir, "train_val_curve_AE.jpg"))


