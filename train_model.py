from util import args, Loss, Trainer, Saver
from radar import MultiRadar
from data import MultiRadarData
from model import ComplexModel

mr = MultiRadar(f0=[60e9, 77e9], K=124.996e12, Nk=[64, 64], Nk_fb=336, fS=2000e3)
d = MultiRadarData(mr, args)

args.dataset = 'dataset_60GHz_77GHz_16384_2048_Nt64.mrd'

if args.dataset == '':
    d.create_dataset_train(args.num_train, args.Nt)
    d.create_dataset_val(args.num_val, args.Nt)
    d.create_dataset_test(args.num_test, args.Nt)
else:
    d.Load(f"./saved/data/" + args.dataset)

if args.checkpoint == '':
    m = ComplexModel(args)
    l = Loss(args)
    t = Trainer(args, data=d, model=m, loss=l)
else:
    s = Saver()
    args, m, l, t = s.Load(args, d, ComplexModel, Loss, Trainer, "./saved/models/" + args.checkpoint)

# train the model
print("Training the model...")
while not t.terminate():
    t.train()

