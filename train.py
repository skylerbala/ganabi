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

    tr_history = model.fit_generator(
        generator=data.naive_generator(trainer.batch_size, 'train'),
        validation_data=data.naive_generator(trainer.batch_size, 'validation'),
        verbose=2,  # one line per epoch
        epochs=trainer.epochs,  # = total data / batch_size
        shuffle=True,
        steps_per_epoch=3,
        validation_steps=2)

    return model


if __name__ == "__main__":
    args = parse_args.parse()
    gin.parse_config_file(args.configpath)
    data = load_data.main(args)
    main(data, args)
