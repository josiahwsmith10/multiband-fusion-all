from util import args, Saver
from data import MultiRadarData
from model import ComplexModel


def main():
    assert args.checkpoint != "", "Please specify a checkpoint to load"

    d = MultiRadarData(args)

    d.create_dataset_train(1, 1)
    d.create_dataset_val(1, 1)
    d.create_dataset_test(args.num_test, args.Nt)

    s = Saver()
    args_in, m, _, _ = s.Load(
        args, d, ComplexModel, None, None, "./saved/models/" + args.checkpoint
    )

    d.test_net(args_in, m)


if __name__ == "__main__":
    main()
