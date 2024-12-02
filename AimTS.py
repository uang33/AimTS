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
    
    # DC
    def _eval_with_pooling_DC(self, x, mask=None, slicing=None, encoding_window=None): 
        out = self.net(x.to(self.device, non_blocking=True))    #  B x T x C'   
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing] 

            out = out.transpose(1, 2)
            out = F.max_pool1d(  
                out,
                kernel_size = out.size(2),
            ).transpose(1, 2)
            
        return out.cpu()
    

    # CI
    def _eval_with_pooling_CI(self, x, mask=None, slicing=None, encoding_window=None): 
        out = self.net(x.to(self.device, non_blocking=True))    #  B x C x T x C'   
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = out.transpose(2, 3)   #  B x C x C' x T
            out = torch.reshape(out, (out.shape[0], out.shape[1]*out.shape[2], out.shape[3]))   #  B x C*C' x T 
            # out = out.transpose(1, 2)
            out = F.max_pool1d(  
                out,
                kernel_size = out.size(2),
            ).transpose(1, 2)
            
        return out.cpu()
    
   
    def finetune_fit(self, train_data, train_labels, test_data, test_labels, finetune_data, epochs, finetune_lr, method,
                     batch_size=None, encoding_window=None, mask=None, ):
                     
        assert self.net is not None, 'please train or load a net first'
        assert train_data.ndim == 3 
        if batch_size is None:
            batch_size = self.batch_size
       
        n_samples, ts_l, C = train_data.shape
        if(finetune_data != 1):
            split = train_test_split(
            train_data, train_labels,
            train_size=finetune_data, random_state=0, stratify=train_labels,shuffle=True
            )
            train_data= split[0]
            train_labels = split[2]
        
        
        label_num1 = len(np.unique(train_labels))
        label_num2 = len(np.unique(test_labels))
        if label_num1 > label_num2:
            label_num2 = label_num1

        # projection head append after encoder
        if method == 'CI':
            self.proj_head = ProjectionHead(input_dims=self.output_dims*C, output_dims=label_num2, hidden_dims=256).to(self.device) # CI
        else:
            self.proj_head = ProjectionHead(input_dims=self.output_dims, output_dims=label_num2, hidden_dims=256).to(self.device) # DC

        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float),
                                      F.one_hot(torch.from_numpy(train_labels).to(torch.long), num_classes=label_num2).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        proj_head_optimizer = torch.optim.AdamW(self.proj_head.parameters(), lr=finetune_lr)
        
        criterion = nn.CrossEntropyLoss()

        epoch_loss_list, iter_loss_list = [], []
        best_acc = 0

        for epoch in range(epochs):
            train_loss = []
            i = 0
            correct = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                if(x.shape[0]==1):
                    continue
                self.net.train()
                self.proj_head.train()
                optimizer.zero_grad()
                proj_head_optimizer.zero_grad()

                if method == 'CI':
                    out = self._eval_with_pooling_CI(x, mask, encoding_window=encoding_window).to(self.device)
                else:
                    out = self._eval_with_pooling_DC(x, mask, encoding_window=encoding_window).to(self.device)
                
                if encoding_window == 'full_series':
                    out = out.squeeze(1)  # B x output_dims
                y_pred = self.proj_head(out).squeeze(1)
                
                loss = criterion(y_pred, y)

                train_loss.append(loss.item())
                y_pred = y_pred.argmax(dim=1)  # (n,)
                y_target = y.argmax(dim=1) # (n,)
                
                correct += (y_pred == y_target).sum().item()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                loss.backward()

                optimizer.step()
                proj_head_optimizer.step()
                iter_loss_list.append(loss.item())
                i += 1

            train_loss = np.average(train_loss)
            epoch_loss_list.append(sum(iter_loss_list) / len(iter_loss_list))
            train_acc = correct / n_samples
            performance = self.finetune_predict(test_data, test_labels, method=method, encoding_window=encoding_window)

            acc = performance['Accuracy']
            print("Epoch: {0} | Train Loss: {1:.3f}, Train Acc: {2:.3f}, Test Acc: {3:.3f}".format(epoch + 1, train_loss, train_acc, acc))

           
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(proj_head_optimizer, epoch + 1, finetune_lr)
               

        return epoch_loss_list, performance
   
    def finetune_predict(self, test_data, test_labels, method, mask=None, encoding_window=None):
        
        label_num2 = len(np.unique(test_labels))

        test_dataset = TensorDataset(torch.from_numpy(test_data).to(torch.float),
                                     F.one_hot(torch.from_numpy(test_labels).to(torch.long), num_classes=label_num2).to(
                                         torch.float))
    
        test_loader = DataLoader(test_dataset, batch_size=8)

        criterion = nn.CrossEntropyLoss()
        org_training = self.net.training
        self.net.eval()
        self.proj_head.eval()

        acc = 0
        correct = 0
        total = test_data.shape[0]

        iter_loss_list = []
        
        with torch.no_grad():

            for index, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                if method == 'CI':
                    out = self._eval_with_pooling_CI(x, mask, encoding_window=encoding_window).to(self.device)
                else:
                    out = self._eval_with_pooling_DC(x, mask, encoding_window=encoding_window).to(self.device)

                if encoding_window == 'full_series':
                    out = out.squeeze(1)  # B x output_dims

                y_pred_prob = self.proj_head(out).squeeze(1)  # (n,n_classes)

                loss = criterion(y_pred_prob, y)
                iter_loss_list.append(loss.item())

                y_pred = y_pred_prob.argmax(dim=1)  # (n,)
                y_target = y.argmax(dim=1) # (n,)
                
                correct += (y_pred == y_target).sum().item()
                
            
        test_loss = (sum(iter_loss_list) / len(iter_loss_list))
               
        acc = correct / total
        metrics_dict = {}
        metrics_dict['Accuracy'] = acc
        metrics_dict['test loss'] = test_loss
        
        self.net.train(org_training)
        self.proj_head.train(org_training)

        return metrics_dict

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


