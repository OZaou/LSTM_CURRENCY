import plotly.graph_objects as go
import pandas as pd
import os
import logging
from utils import ensure_dir

class Visualizer:
    def __init__(self, ticker, config):
        self.ticker = ticker
        self.config = config

    def plot_predictions(self, actuals, predictions):
        plots_dir = self.config['output']['plots_dir']
        ensure_dir(plots_dir)
        df = pd.DataFrame({
            'Actual': actuals,
            'Predicted': predictions
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['Actual'], mode='lines', name='Réel'))
        fig.add_trace(go.Scatter(y=df['Predicted'], mode='lines', name='Prédit'))
        fig.update_layout(
            title=f"Prédiction des prix pour {self.ticker}",
            xaxis_title="Temps",
            yaxis_title="Prix"
        )
        plot_path = os.path.join(plots_dir, f"{self.ticker}_prediction.html")
        fig.write_html(plot_path)
        logging.info(f"Graphique de prédiction sauvegardé à {plot_path}")
