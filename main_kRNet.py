from util import args, Loss, Trainer
from radar import MultiRadar
from data import MultiRadarData
from model import ComplexModel

mr = MultiRadar(f0=[60e9, 77e9], K=124.996e12, Nk=[64, 64], Nk_fb=336, fS=2000e3)
d = MultiRadarData(mr, args)

d.create_dataset_train(2048, 64)
d.create_dataset_val(1024, 64)

m = ComplexModel(args)
l = Loss(args)
t = Trainer(args, data=d, model=m, loss=l)

# train the model
print("Training the model...")
while not t.terminate():
    t.train()