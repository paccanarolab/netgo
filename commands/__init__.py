def predict(args):
    from commands.predict import Predict
    p = Predict(args)
    p.run()


def train(args):
    from commands.train import Train
    t = Train(args)
    t.run()