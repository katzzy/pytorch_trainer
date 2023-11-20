from options.option_interface import prepare_eval_args
from model.model_interface import select_model
from data.augment import transform_eval
from engine.predictor import Predictor


def main():
    args = prepare_eval_args()
    model = select_model(args)
    my_predictor = Predictor(args, model, transform_eval)
    my_predictor.predict_csv()


if __name__ == '__main__':
    main()
