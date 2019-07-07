from utils import parse_args, dir_utils
import load_data
import train
import gin


def main():
    #parse arguments
    args = parse_args.parse()
    args = dir_utils.resolve_run_directory(args)

    gin.parse_config_file(args.configpath)

    #create/load data
    data = load_data.main(args)

    #train model/load model
    model = train.main(data, args)
    print('gang')
    #evaluate model
    # evaluate.main(data, model, args)


if __name__ == "__main__":
    main()
