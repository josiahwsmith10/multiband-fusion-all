import torch

class Saver():
    def __init__(self):
        pass
    
    def Save(self, args, model, loss, trainer, PATH):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': loss,
            'args': args,
            'train_loss': model.train_loss,
            'val_loss': model.val_loss
        }
        
        torch.save(checkpoint, PATH)
        print(f"Saved model to: {PATH}")
    
    def Load(self, data, ModelClass, TrainerClass, PATH, batch_size=None):
                
        checkpoint = torch.load(PATH)
        
        args = checkpoint['args']
        args.batch_size = batch_size if not batch_size is None else args.batch_size
        model = ModelClass(args)
        loss = checkpoint['loss']
        trainer = TrainerClass(args, data, model, loss)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
            model.train_loss = checkpoint['train_loss']
            model.val_loss = checkpoint['val_loss']
            
            for tl, vl in zip(model.train_loss, model.val_loss):
                trainer.liveloss.update({'log loss': tl, 'val_log loss': vl})
        
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        model.eval()
        print(f"Loaded model from: {PATH}")
        return args, model, loss, trainer
    
    def LoadModel(self, ModelClass, PATH):
        checkpoint = torch.load(PATH)
        
        args = checkpoint['args']
        model = ModelClass(args)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        print(f"Loaded model from: {PATH}")
        return args, model