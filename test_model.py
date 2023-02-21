from util import args, Saver
from radar import MultiRadar
from data import MultiRadarData
from model import ComplexModel

assert args.checkpoint != '', "Please specify a checkpoint to load"

mr = MultiRadar(f0=[60e9, 77e9], K=124.996e12, Nk=[64, 64], Nk_fb=336, fS=2000e3)
d = MultiRadarData(mr, args)

d.create_dataset_train(1, 1)
d.create_dataset_val(1, 1)
d.create_dataset_test(args.num_test, args.Nt)

s = Saver()
args, m, _, _ = s.Load(args, d, ComplexModel, None, None, "./saved/models/" + args.checkpoint)

d.test_net(args, m)