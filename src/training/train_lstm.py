from src.models.lstm_model import build_lstm
from src.training.rnn_common import parse_args, train_rnn

OUT_DIR = "models/lstm"


def main():
    args = parse_args("LSTM")
    train_rnn(
        build_lstm, args.out_dir or OUT_DIR, "LSTM",
        map_emoticons=args.map_emoticons, epochs=args.epochs, glove_path=args.glove_path,
        results_path=args.results_path, subset=args.subset,
    )


if __name__ == "__main__":
    main()
