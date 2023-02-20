from util import args, Loss, Trainer, Saver
from radar import MultiRadar
from data import MultiRadarData
from model import ComplexModel

mr = MultiRadar(f0=[60e9, 77e9], K=124.996e12, Nk=[64, 64], Nk_fb=336, fS=2000e3)
d = MultiRadarData(mr, args)

d.create_dataset_train(64, 64)
d.create_dataset_val(64, 64)
d.create_dataset_test(64, 4)

s = Saver()
args, m, l, t = s.Load(args, d, ComplexModel, Trainer, "./saved/models/" + args.checkpoint)

d.test_net(args, m, d)