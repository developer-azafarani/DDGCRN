import torch
import os
import time
from lib.metrics import MAE_torch, MSE_torch, RMSE_torch  # Assume from your lib

class Trainer:
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler=None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.args.device)
            target = target.to(self.args.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / self.train_per_epoch

    def val_epoch(self, epoch):
        # Stub: similar to train_epoch but eval mode
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                output = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()
        return total_val_loss / self.val_per_epoch

    def train(self):
        min_val_loss = float('inf')
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), self.args.log_dir + "best_model.pth")
            print(f"Epoch {epoch}: Train Loss {train_loss}, Val Loss {val_loss}")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def test(self, model, args, test_loader, scaler, logger=None):
        model.eval()
        with torch.no_grad():
            mae, mse, rmse = 0, 0, 0
            for data, target in test_loader:
                data = data.to(args.device)
                target = target.to(args.device)
                output = model(data)
                mae += MAE_torch(output, target, 0.0).item()
                mse += MSE_torch(output, target).item()
                rmse += RMSE_torch(output, target).item()
        print(f"Test: MAE {mae / len(test_loader)}, MSE {mse / len(test_loader)}, RMSE {rmse / len(test_loader)}")

class TrainerExtended(Trainer):
    def train(self):
        total_epochs = self.args.epochs
        stage_a_epochs = int(total_epochs * 0.3)  # 30% for Stage A
        stage_b_epochs = int(total_epochs * 0.3)  # 30% for Stage B
        stage_c_epochs = total_epochs - stage_a_epochs - stage_b_epochs

        # Stage A: Normal pretrain (bias to Ad, Al, Ag; filter low-resid samples if possible)
        print("Stage A: Normal pretrain")
        self.model.train()
        for epoch in range(stage_a_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            print(f"Stage A Epoch {epoch}: Train {train_loss}, Val {val_loss}")

        # Stage B: Abnormal specialization (bias to Adelta; higher LR, focus on high-resid)
        print("Stage B: Abnormal specialization")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 2  # Higher LR
        for epoch in range(stage_a_epochs, stage_a_epochs + stage_b_epochs):
            train_loss = self.train_epoch(epoch)  # Optionally filter high-resid batches
            val_loss = self.val_epoch(epoch)
            print(f"Stage B Epoch {epoch}: Train {train_loss}, Val {val_loss}")

        # Stage C: Joint fine-tune (all graphs, full losses: MAE + sparsity + smoothness)
        print("Stage C: Joint fine-tune")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr_init * 0.1  # Lower LR
        min_val_loss = float('inf')
        for epoch in range(stage_a_epochs + stage_b_epochs, total_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                # Full losses
                main_loss = self.loss(output, target)
                sparsity_loss = torch.norm(self.model.generate_abnormality_graph(data)[0], p=1) / self.num_nodes**2  # L1 on Adelta
                smoothness_loss = torch.mean((output[:, 1:] - output[:, :-1])**2)  # Temporal smoothness
                contrastive_loss = 0  # Optional: add embedding contrastive
                loss = main_loss + 0.1 * sparsity_loss + 0.1 * smoothness_loss + contrastive_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / self.train_per_epoch
            val_loss = self.val_epoch(epoch)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), self.args.log_dir + "best_model.pth")
            print(f"Stage C Epoch {epoch}: Train {train_loss}, Val {val_loss}")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
