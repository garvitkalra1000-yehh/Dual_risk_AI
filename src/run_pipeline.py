from src import build_outputs, clean, explain, features, generate_data, priority, score, segment, train


def main() -> None:
    generate_data.main()
    clean.main()
    features.main()
    score.main()
    train.main()
    explain.main()
    segment.main()
    priority.main()
    build_outputs.main()


if __name__ == "__main__":
    main()
