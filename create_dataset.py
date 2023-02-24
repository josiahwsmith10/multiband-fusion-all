from util import args
from data import MultiRadarData

d = MultiRadarData(args)

d.create_dataset_train(args.num_train, args.Nt)
d.create_dataset_val(args.num_val, args.Nt)
d.create_dataset_test(args.num_test, args.Nt)

d.Save(f"./saved/data/dataset_{args.dataset_type}_{args.num_train}_{args.num_val}_Nt{args.Nt}.mrd")

