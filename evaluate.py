from utils import parse_args
import gin
import random
import load_data
import train


# @gin.configurable
# class Evaluator(object):
#     def __init__(self):
#
#     def eval():
#
#     def _single_move():
#
#
# # TODO


def main(args, data, model):
    agent = random.choice(list(data.test_data))
    steps = sum([len(game[0]) for game in data.test_data[agent]])

    print('\n# Evaluate on test data')
    results = model.evaluate_generator(
        generator=data.naive_generator(32, 'test'),
        steps=steps
    )
    print('test loss, test acc:', results)

    # read in test data
    # forward pass with model
    # display metrics, save results


if __name__ == "__main__":
    args = parse_args.parse()
    data = load_data.main()
    model = train.main(data, args)
    main(args, data, model)
