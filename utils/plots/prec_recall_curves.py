from sklearn import metrics
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt


# def plot_curve_testing(y_test, y_pred_proba):
#     """
#     #Построение roc_auc и precision_recall кривых на тестовых скважинах
#     """
#
#     # well_id = [int(well['well id'].unique()[0]) for well in test_wells]
#     colors = ['red', 'blue', 'green', 'black', 'orange']
#
#     fig = make_subplots(rows=1, cols=2,
#                         subplot_titles=['$\Large\\textbf{ROC-curve}$', '$\Large\\textbf{Precision-recall-curve}$'])
#
#     # for y_final, final_proba, well, col in tqdm(zip(y_finals, pred_finals_proba, well_id, colors),
#     #                                             desc='Plotting_test_curve...', total=len(test_wells)):
#     fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
#     prec, rec, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
#
#     fig.add_trace(
#         go.Scatter(x=fpr, y=tpr, marker_color=colors[0], name=f'Well_1, \t S_roc={round(metrics.auc(fpr, tpr), 3)}'),
#         1, 1)
#     fig.add_trace(go.Scatter(x=rec, y=prec, marker_color=colors[0],
#                              name=f'Well_1, \t S_pr={round(metrics.auc(rec, prec), 3)}'), 1, 2)
#
#     fig.update_layout(plot_bgcolor='white', height=700, width=1450,
#                       xaxis=dict(title=dict(text='False Positive Rate', font=dict(size=20)), tickfont=dict(size=20),
#                                  gridcolor='lightgrey', \
#                                  zerolinecolor='lightgrey', constrain='domain'), \
#                       yaxis=dict(title=dict(text='True Positive Rate', font=dict(size=20)), tickfont=dict(size=20),
#                                  gridcolor='lightgrey', \
#                                  zerolinecolor='lightgrey'),
#                       xaxis2=dict(title=dict(text='Recall', font=dict(size=20)), tickfont=dict(size=20),
#                                   gridcolor='lightgrey', zerolinecolor='lightgrey'),
#                       yaxis2=dict(title=dict(text='Precision', font=dict(size=20)), tickfont=dict(size=20),
#                                   gridcolor='lightgrey', zerolinecolor='lightgrey'))
#
#     fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='black'), showlegend=False), 1, 1)
#     fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], line=dict(dash='dash', color='black'), showlegend=False), 1, 2)
#
#     pio.write_image(fig, 'ROC-PR-curve_test.jpg', scale=3)

def plot_curve_testing(y_test, y_pred_proba):
    precision = dict()
    recall = dict()
    average_precision = dict()

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_pred_proba.ravel()
    )
    average_precision["micro"] = average_precision_score(y_test, y_pred_proba, average="micro")
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot()
    _ = display.ax_.set_title("Micro-averaged over all classes")
    plt.show()
    a = 1
