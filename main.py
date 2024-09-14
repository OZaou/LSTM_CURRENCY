
import torch
import logging
from utils import setup_logging, load_config, ensure_dir
from data_loader import ForexDataset
from model import ForexModel
from trainer import Trainer
from visualizer import Visualizer
import multiprocessing

def train_currency(ticker, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ForexDataset(
        ticker=ticker,
        start_date=config['data']['start_date'],
        interval=config['data']['interval'],
        features=config['data']['features'],
        time_step=config['data']['time_step'],
        test_size=config['data']['test_size'],
        validation_size=config['data']['validation_size']
    )
    dataset.load_data()
    dataset.preprocess_data()

    input_size = dataset.get_input_size()
    config['model']['input_size'] = input_size

    model = ForexModel(
        input_size=input_size,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    ).to(device)

    trainer = Trainer(model, dataset, config, device)
    trainer.train()
    predictions, actuals = trainer.test()

    visualizer = Visualizer(ticker, config)
    visualizer.plot_predictions(actuals, predictions)

if __name__ == "__main__":
    config = load_config('config.yaml')
    setup_logging(
        log_file=config['logging']['log_file'],
        level=getattr(logging, config['logging']['level'])
    )
    tickers = config['data']['tickers']

    processes = []
    for ticker in tickers:
        p = multiprocessing.Process(target=train_currency, args=(ticker, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
