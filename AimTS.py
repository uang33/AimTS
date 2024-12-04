import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.encoder import TSEncoder, ProjectionHead
from models.losses import  *
from models.augmentation import aug
from utils import  adjust_learning_rate,tao
import os
from torch import nn
from sklearn.model_selection import train_test_split
import os

class AimTS():
    '''The AimTS model'''
    
    def __init__(
        self,
        input_dims=1,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=64,
        max_train_length=None,
        alpha=0.5,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
        method='CI',

    ):
        ''' Initialize an AimTS model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.

        '''
        
        super().__init__()
        self.output_dims = output_dims
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.inputdims = input_dims
        self.max_train_length = max_train_length
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.ts_encoder = TSEncoder(input_dims=1, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)    # CI
       
        self.proj_head_ts = ProjectionHead(input_dims=output_dims, output_dims=128, hidden_dims=256).to(self.device)   
        self.proj_head_img = ProjectionHead(input_dims=output_dims, output_dims=128, hidden_dims=256).to(self.device)  
        self.proj_head_pro = ProjectionHead(input_dims=output_dims, output_dims=128, hidden_dims=256).to(self.device)
        self.proj_head_aug =ProjectionHead(input_dims=output_dims, output_dims=128, hidden_dims=256).to(self.device)
    
        self.net = torch.optim.swa_utils.AveragedModel(self.ts_encoder)
        self.net.update_parameters(self.ts_encoder)
   
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        self.n_epochs = 0
        self.n_iters = 0
        self.method = method

    def fit(self,  data_loader, n_epochs=None, n_iters=None):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''

        para_num = sum(p.numel() for p in self.ts_encoder.parameters())
        print(f"Number of parameters: {para_num}")

        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Identity()
        resnet18 = resnet18.to(self.device)
       
       # parameters
        ts_parms = self.ts_encoder.parameters()
        proj_img_parms = self.proj_head_img.parameters()
        proj_ts_parms = self.proj_head_ts.parameters()
        pro_cl_parms = self.proj_head_pro.parameters()
        aug_cl_parms = self.proj_head_aug.parameters()
        
        ts2img_parms = list(ts_parms) + list(proj_img_parms) + list(proj_ts_parms)
        aug_parms = list(ts_parms)+list(pro_cl_parms) +list(aug_cl_parms)

        # optimizer
        ts2img_optimizer = torch.optim.AdamW(ts2img_parms, lr=self.lr)
        aug_optimizer = torch.optim.AdamW(aug_parms, lr=self.lr)

        loss_log = []
        iters = 0

        train_loader = data_loader
        flag = 0  
        if flag == 0:
            print('>>>>>>>warm up : >>>>>>>>>>>>>>>>>>>>>>>>>>')

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            if self.n_epochs == 1 and flag == 0:
                print('>>>>>>>knowledge distillation : >>>>>>>>>>>>>>>>>>>>>>>>>>')
                flag = 1
                self.n_epochs -= 1
                train_loss = []
                loss_log = []
                ts_parms = self.ts_encoder.parameters()
                proj_ts_parms = self.proj_head_ts.parameters()
                ts2img_parms = list(ts_parms) + list(proj_ts_parms)
                ts2img_optimizer = torch.optim.AdamW(ts2img_parms, lr=self.lr)

            train_loss = []
            train_loss_img = []
            train_loss_proto = []
            train_loss_ins = []
            train_loss_aug = []
            n_epoch_iters = 0
            
            interrupted = False
            for i, (batch, label, img) in enumerate(train_loader):
                    
                B = batch.size(0)
                # print(B)
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]    # 裁到max_train_length长
                x = x.to(self.device)
                label = label.to(self.device)
                img = img.to(self.device)
            
                del label

                
                ts2img_optimizer.zero_grad()
                aug_optimizer.zero_grad()

                # ts2img
                ts_embedding = self.ts_encoder(x)
                ts_embedding = ts_embedding.mean(1)   # B x T x C'
                ts_embedding = ts_embedding.permute(0, 2, 1)
                ts_embedding = F.max_pool1d(  
                        ts_embedding,
                        kernel_size = ts_embedding.size(2),
                    )   # B x C'
                ts_embedding = ts_embedding.squeeze(-1)
                ts_embedding = self.proj_head_ts(ts_embedding)

                img_embedding = resnet18(img)
                img_embedding = self.proj_head_img(img_embedding)

                loss_img = img_loss(ts_embedding, img_embedding)

                # knowledge distillation and augmentation
                if flag == 1:
                   
                    # multi aug
                    aug1_all = aug(x)    # list G 
                    aug2_all = aug(x)    
                    
                    out1_list, out2_list = [], []
                    ts_list1, ts_list2=[], []
                    for j in range(len(aug1_all)):
                        x1 = aug1_all[j]
                        x2 = aug2_all[j]
                        ts_list1.append(x1)
                        ts_list2.append(x2)
                        out1 = self.ts_encoder(x1.clone())  


                        out1 = out1.mean(1)   # B x T x C'
                        out2 = self.ts_encoder(x2.clone())
                        out2 = out2.mean(1)   # B x T x C'
                        out1_list.append(out1)
                        out2_list.append(out2)


                    tao_aug = tao(ts_list1, ts_list2, method='+-')            #负样本对的权重
                    tao_aug = tao_aug.to(self.device)

                    del ts_list1, ts_list2

                    out1 = torch.stack(out1_list, 0)   # G x B x T x C'         #G=5 x B x T x C'
                    out2 = torch.stack(out2_list, 0)    

                    loss_ins, loss_aug = ProtoLoss(out1, out2, tao_aug, self.proj_head_pro, self.proj_head_aug)
                    train_loss_ins.append(loss_ins.item())
                    train_loss_aug.append(loss_aug.item())

                    
                    loss_proto = 0.9 * loss_ins + 0.1 * loss_aug
                   
                    del out1, out2, out1_list, out2_list, x1, x2, aug1_all, aug2_all
                    train_loss_proto.append(loss_proto.item())
                    train_loss_img.append(loss_img.item())
                    loss = loss_img + loss_proto
                    

                else:   
                    loss = loss_img

                train_loss.append(loss.item())
                
                if (i + 1) % 1000 == 0:
                    if flag == 1:
                        
                        print("\titers: {0}, epoch: {1} | Train loss: {2:.7f}, img loss: {3:.7f}, proto loss: {4:.7f}, pro loss: {5:.7f}, aug loss: {6:.7f}".format(
                            i + 1, self.n_epochs + 1, loss.item(), loss_img.item(), loss_proto.item(),loss_ins.item(),loss_aug.item()))
                        
                    else:
                        print("\titers: {0}, epoch: {1} | Train loss: {2:.7f}".format(
                            i + 1, self.n_epochs + 1, loss.item()))

                loss.backward()

                aug_optimizer.step()
                ts2img_optimizer.step()
              
                if flag == 1:
                    self.net.update_parameters(self.ts_encoder)
                        
                iters += 1

                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

                torch.cuda.empty_cache()
            
            if interrupted:
                break
            
            if flag == 1:
                train_loss_proto = np.average(train_loss_proto)

            train_loss = np.average(train_loss)   

            train_loss_aug = np.average(train_loss_aug)
            loss_log.append(train_loss)

            if flag == 1:
                print("Epoch: {0} | Train Loss: {1:.3f}, img loss: {2:.3f}, proto loss: {3:.3f}".format(
                    self.n_epochs + 1, train_loss, train_loss-train_loss_proto, train_loss_proto))
            else:
                print("Epoch: {0} | Train Loss: {1:.3f}".format(
                    self.n_epochs + 1, train_loss))

            # figure
            if flag == 1:
                with open("./pic/pretrain/ts2img_train_loss.txt", 'w') as train_los:
                    train_los.write(str(loss_log))
                train_los.close()

            if (self.n_epochs + 1) % 10 == 0 and flag == 1:
                adjust_learning_rate(ts2img_optimizer, self.n_epochs + 1, self.lr)
                adjust_learning_rate(aug_optimizer, self.n_epochs + 1, self.lr)

            self.n_epochs += 1
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, train_loss)
            
            torch.cuda.empty_cache()

        return loss_log
    
    

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        pretrain_dict = torch.load(fn, map_location=self.device)
        finetune_dict = self.net.state_dict()
        pretrain_dict_stay = {k: v for k, v in pretrain_dict.items() if k in finetune_dict}
        finetune_dict.update(pretrain_dict_stay)
        self.net.load_state_dict(finetune_dict)


