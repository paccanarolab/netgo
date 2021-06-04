def predict(args):
    from commands.predict import Predict
    p = Predict(args)
    p.run()


def train(args):
    from commands.train import Train
    t = Train(args)
    t.run()


def filter_gaf(args):
    from commands.filter_gaf import FilterGAF
    f = FilterGAF(args)
    f.run()


def filter_string(args):
    from commands.filter_string import FilterSTRING
    f = FilterSTRING(args)
    f.run()


def map_string(args):
    from commands.map_string import MapSTRING
    m = MapSTRING(args)
    m.run()