
import torch

class Helper:
    def __init__(self):
        pass

    #Function to save checkpoints
#To allow you to resume training from the last saved point, avoiding the need to start over if something interrupts the training. This helps save time and effort.
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, file_path='checkpoint.pth'):
        checkpoint = {
            'model_state_dict': model.state_dict(),#state_dict() is a dictionary that stores all the learnable parameters (like weights and biases) and buffers of a model or optimizer in PyTorch.
            'optimizer_state_dict': optimizer.state_dict(),#optimizer.state_dict() stores the current state of the optimizer, including parameters like learning rates and momentum, allowing you to resume training with the same optimization settings.
            'epoch': epoch,#Saves the current training epoch (the number of times the model has seen the entire dataset).
        }
        
        torch.save(checkpoint, file_path)
        print(f'Checkpoint saved at {file_path}')

    #Function to load checkpoints
    @staticmethod
    def load_checkpoint(model, optimizer=None, file_path='checkpoint.pth'):
        print(file_path)
        checkpoint = torch.load(file_path)
        print("here")
        # print(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("here working......")
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        
        print(f'Checkpoint loaded from {file_path} (Epoch: {epoch})')
        
        return model, optimizer, epoch
    @staticmethod
    def transfer_learning(model,model_name,file_path,device):
        try:

            checkpoint = torch.load(file_path)

            # Step 1: Modify the classification layer for transfer learning
            if model_name=='InceptionResnetV1':
                model.logits = nn.Linear(in_features=model.logits.in_features,out_features=len(dataset.classes) , bias=True)
        
            elif model_name=='VIT':
                model.linear_head = nn.Linear(in_features=model.linear_head.in_features, out_features=len(dataset.classes), bias=True)
                # Remove the `linear_head` weights from the checkpoint to avoid size mismatch
                del checkpoint['model_state_dict']['linear_head.weight']
                del checkpoint['model_state_dict']['linear_head.bias']
                
            # Load the remaining weights
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(device)
            return model
        except Exception as ex:
            print(f"No previous learning found. Training model from scratch {ex}")
            return model
    @staticmethod
    def get_checkpoint_path(key):
        return f'checkpoint_{key}.pth'