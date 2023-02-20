from util import args
from radar import MultiRadar
from data import MultiRadarData

mr = MultiRadar(f0=[60e9, 77e9], K=124.996e12, Nk=[64, 64], Nk_fb=336, fS=2000e3)
d = MultiRadarData(mr, args)

d.create_dataset_train(args.num_train, args.Nt)
d.create_dataset_val(args.num_val, args.Nt)
d.create_dataset_test(args.num_test, args.Nt)

d.Save(f"./saved/data/dataset_60GHz_77GHz_{args.num_train}_{args.num_val}_Nt{args.Nt}.mrd")