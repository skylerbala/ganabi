from utils import parse_args, dir_utils
import load_data
import train
import gin
import evaluate


def main():
    #parse arguments
    args = parse_args.parse()
    args = dir_utils.resolve_run_directory(args)

    gin.parse_config_file(args.configpath)

    data = load_data.main(args)

    model = train.main(data, args)

    evaluate.main(args, data, model)


if __name__ == "__main__":
    main()
