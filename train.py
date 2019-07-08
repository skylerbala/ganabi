import random

from utils import parse_args
import importlib
import load_data
import gin
import sys


@gin.configurable
class Trainer(object):
    def __init__(self,
                 args,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 batch_size=None,
                 epochs=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs


def main(data, args):
    trainer = Trainer(args)  # gin configured

    # FIXME: combine into one line once stuff works
    sys.path.insert(0, args.modedir)
    mode_module = importlib.import_module(args.mode)
    model = mode_module.build_model(args)

    model.compile(
        optimizer=trainer.optimizer,
        loss=trainer.loss,
        metrics=trainer.metrics)

    agent = random.choice(list(data.train_data))

    steps_per_epoch = sum([len(game[0]) for game in data.train_data[agent]]) / 4
    validation_steps = sum([len(game[0]) for game in data.validation_data[agent]]) / 4

    print('# Fit model on training data')
    history = model.fit_generator(
        generator=data.naive_generator(trainer.batch_size, 'train'),
        steps_per_epoch=steps_per_epoch,
        verbose=2,  # one line per epoch
        epochs=trainer.epochs,  # = total data / batch_size
        validation_data=data.naive_generator(trainer.batch_size, 'validation'),
        validation_steps=validation_steps,
        shuffle=True)
    print('\nhistory dict:', history.history)

    return model


if __name__ == "__main__":
    args = parse_args.parse()
    gin.parse_config_file(args.configpath)
    data = load_data.main(args)
    main(data, args)
