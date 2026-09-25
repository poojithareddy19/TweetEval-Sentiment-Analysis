from src.models.lstm_model import build_lstm
from src.training.rnn_common import parse_args, train_rnn

OUT_DIR = "models/lstm"


def main():
    args = parse_args("LSTM")
    train_rnn(build_lstm, OUT_DIR, "LSTM", map_emoticons=args.map_emoticons, epochs=args.epochs)


if __name__ == "__main__":
    main()
