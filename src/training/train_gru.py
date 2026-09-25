from src.models.gru_model import build_gru
from src.training.rnn_common import parse_args, train_rnn

OUT_DIR = "models/gru"


def main():
    args = parse_args("GRU")
    train_rnn(
        build_gru, args.out_dir or OUT_DIR, "GRU",
        map_emoticons=args.map_emoticons, epochs=args.epochs, glove_path=args.glove_path,
        results_path=args.results_path, subset=args.subset,
    )


if __name__ == "__main__":
    main()
