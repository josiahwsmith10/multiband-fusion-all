from util import args, Loss, Trainer, Saver
from data import MultiRadarData
from model import ComplexModel


def main():
    d = MultiRadarData(args)

    if args.dataset == "":
        d.create_dataset_train(args.num_train, args.Nt)
        d.create_dataset_val(args.num_val, args.Nt)
        d.create_dataset_test(args.num_test, args.Nt)
    else:
        d.Load(f"./saved/data/" + args.dataset)

    if args.checkpoint == "":
        m = ComplexModel(args)
        l = Loss(args)
        t = Trainer(args, data=d, model=m, loss=l)
    else:
        s = Saver()
        args, m, l, t = s.Load(
            args, d, ComplexModel, Loss, Trainer, "./saved/models/" + args.checkpoint
        )

    # train the model
    print("Training the model...")
    while not t.terminate():
        t.train()


if __name__ == "__main__":
    main()
