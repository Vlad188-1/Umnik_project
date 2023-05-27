import plotly.graph_objects as go
import plotly.offline
import plotly.io as pio
from plotly.subplots import make_subplots

from pandas import DataFrame, Series
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class Plots:
    #
    # @staticmethod
    # def plotDistributionNanValues(data: DataFrame):
    #
    #     dict_nan_values = {}
    #     for column in data.columns:
    #         dict_nan_values[column] = data[column].isna().sum()
    #     dict_nan_values = dict(sorted(dict_nan_values.items(), key=lambda item: item[1]))
    #
    #     # Plot graphic
    #     fig = go.Figure([go.Bar(x=list(dict_nan_values.keys()),
    #                             y=list(dict_nan_values.values()),
    #                             )])
    #     fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
    #                       marker_line_width=1.5, opacity=0.6)
    #     fig.update_layout(plot_bgcolor="white",
    #                       xaxis=dict(title=dict(text="<b>Сolumn names", font=dict(size=16)), showgrid=True),
    #                       yaxis=dict(showgrid=True),
    #                       title=dict(text="The number of Nan values", font=dict(size=24)))
    #     fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
    #     fig.update_yaxes(showgrid=True, gridcolor='lightgrey')
    #
    #     plotly.offline.plot(fig, filename='plots/Nan.html', auto_open=True)
    @staticmethod
    def plotDistributionNanValues(data: DataFrame):

        dict_nan_values = {}
        for column in data.columns:
            dict_nan_values[column] = data[column].isna().sum()
        dict_nan_values = dict(sorted(dict_nan_values.items(), key=lambda item: item[1]))

        keys = list(dict_nan_values.keys())
        values = list(dict_nan_values.values())
        mean_value = int(sum(list(dict_nan_values.values())) / len(dict_nan_values))
        big_value = int(mean_value * 1.2)

        profit_color = [{p < mean_value: 'green', mean_value <= p <= big_value: 'orange', p > big_value: 'red'}[True]
                        for p in values]

        plt.figure(figsize=(16, 14))
        plt.title("Количество NaN значений для каждой переменной", fontweight="bold")
        plt.bar(keys, values, color=profit_color)
        plt.show()
    #
    # @staticmethod
    # def plotDistributedTargetVariable(y):
    #     from collections import Counter
    #     from numpy import ndarray
    #     if isinstance(y, Series):
    #         count_values = y.value_counts().to_dict()
    #     elif isinstance(y, ndarray):
    #         count_values = Counter(y)
    #     name_classes, count_classes = count_values.keys(), count_values.values()
    #
    #     fig = go.Figure([go.Bar(x=list(name_classes),
    #                             y=list(count_classes),
    #                             )])
    #     fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
    #                       marker_line_width=1.5, opacity=0.6)
    #     fig.update_layout(plot_bgcolor="white",
    #                       xaxis=dict(title=dict(text="<b>Name classes", font=dict(size=16)), showgrid=True),
    #                       yaxis=dict(showgrid=True),
    #                       title=dict(text="Distribution target variable", font=dict(size=24)))
    #     fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
    #     fig.update_yaxes(showgrid=True, gridcolor='lightgrey')
    #     plotly.offline.plot(fig, filename="plots/TargetVariable.html", auto_open=True)

    @staticmethod
    def plotDistributedTargetVariable(y):
        from collections import Counter
        from numpy import ndarray
        if isinstance(y, Series):
            count_values = y.value_counts().to_dict()
        elif isinstance(y, ndarray):
            count_values = Counter(y)
        name_classes, count_classes = list(count_values.keys()), list(count_values.values())

        plt.figure(figsize=(16, 14))
        plt.title("Распределение целевой переменной", fontweight="bold")

        mean_value = int(sum(list(count_values.values())) / len(count_values))
        big_value = int(mean_value * 1.2)
        profit_color = [{p < mean_value: 'red', mean_value <= p <= big_value: 'orange', p > big_value: 'green'}[True]
                        for p in count_classes]
        plt.bar(name_classes, count_classes, color=profit_color)
        plt.show()


def plotTrainValidCurve(history: dict, out_dir: str):
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

    fig.update_layout(plot_bgcolor='white', title=dict(text='<b>Обучение и валидация нейронной сети', x=0.5,
                                                       font=dict(size=20)),
                      legend=dict(font=dict(size=14)), height=550, width=1150, showlegend=True)
    fig.update_xaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey',
                     title=dict(text='Epochs', font=dict(size=23)))
    fig.update_xaxes(row=1, col=2, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey',
                     title=dict(text='Epochs', font=dict(size=23)))

    fig.update_yaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')
    fig.update_yaxes(row=1, col=2, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')

    # fig.write_image(Path(out_dir, "train_val_curve.jpg"))
    pio.write_image(fig, Path(out_dir, "train_val_curve.jpg"), scale=3, width=1200, height=600)


def plotTrainValidCurveAE(history: dict, out_dir: str):

    loss = history['train_loss']
    val_loss = history['val_loss']

    epochs = np.arange(1, len(loss) + 1)

    # Plot loss and accuracy curves
    fig = make_subplots(rows=1, cols=1, subplot_titles=['$\Large\\textbf{Loss}$', '$\Large\\textbf{Loss}$'],
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', showlegend=False, marker_color='blue'), 1, 1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', showlegend=False, marker_color='red'), 1, 1)

    fig.update_layout(plot_bgcolor='white', title=dict(text='<b>Обучение и валидация нейронной сети', x=0.5,
                                                       font=dict(size=20)),
                      legend=dict(font=dict(size=14)), height=550, width=1150, showlegend=True)
    fig.update_xaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey',
                     title=dict(text='Epochs', font=dict(size=23)))

    fig.update_yaxes(row=1, col=1, nticks=10, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')

    # fig.write_image(Path(out_dir, "train_val_curve_AE.jpg"))
    pio.write_image(fig, Path(out_dir, "train_val_curve_AE.jpg"), scale=3, width=1200, height=600)


