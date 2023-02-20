from util import args
from radar import MultiRadar
from data import MultiRadarData

mr = MultiRadar(f0=[60e9, 77e9], K=124.996e12, Nk=[64, 64], Nk_fb=336, fS=2000e3)
d = MultiRadarData(mr, args)

d.create_dataset_train(1048576, 64)
d.create_dataset_val(2048, 64)

d.Save(".\saved\data\data60GHz_77GHz_1048576_2048_Nt64.drd")