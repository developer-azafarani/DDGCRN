import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class D3MG_3DP(nn.Module):
    def __init__(self, args):
        super(D3MG_3DP, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        self.input_dim = 3  # flow + day + week
        self.embed_dim = args.embed_dim
        self.rnn_units = args.rnn_units
        self.num_layers = args.num_layers
        self.grid_size = math.ceil(math.sqrt(self.num_nodes))
        self.padded_nodes = self.grid_size ** 2
        self.grid_h = self.grid_size
        self.grid_w = self.grid_size
        print(f"Padding nodes from {self.num_nodes} to {self.padded_nodes} for grid {self.grid_h}x{self.grid_w}")

        # Base components for normal path
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim))
        self.encoder = nn.ModuleList([DMGCGRUCell(self.rnn_units, self.input_dim, args) for _ in range(self.num_layers)])
        self.decoder = nn.ModuleList([DMGCGRUCell(self.rnn_units, self.input_dim, args) for _ in range(self.num_layers)])

        # Abnormal path (decomposition as per paper)
        self.encoder_ab = nn.ModuleList([DMGCGRUCell(self.rnn_units, self.input_dim, args) for _ in range(self.num_layers)])
        self.decoder_ab = nn.ModuleList([DMGCGRUCell(self.rnn_units, self.input_dim, args) for _ in range(self.num_layers)])

        self.output_layer = nn.Linear(self.rnn_units, args.output_dim)

        # 3D Conv for CPT views
        self.conv3d = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.res_unit = nn.Sequential(nn.Conv3d(32, 32, (1, 1, 1)), nn.ReLU())

        # CPT Fusion (adjusted for [B, horizon, N, D])
        self.cpt_fusion = CPTFusion(args.output_dim)

        # Better initialization to stabilize initial losses
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y=None, batches_seen=None):
        B, T_lag, N, D = x.shape

        # Step 1: Extract CPT views from flow channel (fix: all from flow, time features used separately)
        flow = x[..., 0]  
        closeness = flow[:, -self.args.lag:] 
        period = self.extract_period(flow) 
        trend = self.extract_trend(flow) 
        views = [closeness, period, trend]

        # Process views to grid_features
        grid_features = []
        for v in views:
            pad = torch.zeros(B, T_lag, self.padded_nodes - N, device=x.device)
            v_padded = torch.cat([v, pad], dim=-1)  
            v_grid = v_padded.view(B, 1, T_lag, self.grid_h, self.grid_w)  
            h = F.relu(self.conv3d(v_grid))  
            r = self.res_unit(h) + h
            r_flat = r.view(B, 32, T_lag, -1)[:, :, :, :N]  
            r_trans = r_flat.permute(0, 3, 1, 2).reshape(B, N, 32 * T_lag)  
            grid_features.append(r_trans)

        # Step 2: Generate graphs
        Ad = self.generate_distance_graph(x)
        Al = self.generate_latent_graph(x)
        Ag = self.generate_grid_graph(grid_features)
        Adelta, resid_stats = self.generate_abnormality_graph(x)
        A_list = [self.region_divide(A) for A in [Ad, Al, Ag, Adelta]]

        # Step 3: Decomposition for normal and abnormal (per paper)
        flow_mean = flow.mean(dim=1, keepdim=True)  
        resid = flow - flow_mean  
        repeated_mean = flow_mean.repeat(1, T_lag, 1).unsqueeze(-1)  
        time_feats = x[..., 1:]  
        normal_input = torch.cat([repeated_mean, time_feats], dim=-1)  
        resid = resid.unsqueeze(-1)  
        abnormal_input = torch.cat([resid, time_feats], dim=-1)  

        # Encoder normal
        init_state = torch.zeros(B, N, self.rnn_units, device=x.device)
        hidden_normal = [init_state] * self.num_layers
        for t in range(T_lag):
            for layer in range(self.num_layers):
                hidden_normal[layer] = self.encoder[layer](normal_input[:, t], hidden_normal[layer], A_list, resid_stats)

        # Decoder normal
        outputs_normal = []
        go = torch.zeros(B, N, self.input_dim, device=x.device)
        dec_input = go
        for t in range(self.args.horizon):
            for layer in range(self.num_layers):
                hidden_normal[layer] = self.decoder[layer](dec_input, hidden_normal[layer], A_list, resid_stats)
            out = self.output_layer(hidden_normal[-1])  
            outputs_normal.append(out)
            time_feat = x[:, -1, :, 1:] if y is None else y[:, t, :, 1:]  
            dec_input = torch.cat([out, time_feat], dim=-1)  

        # Encoder abnormal
        hidden_ab = [init_state] * self.num_layers
        for t in range(T_lag):
            for layer in range(self.num_layers):
                hidden_ab[layer] = self.encoder_ab[layer](abnormal_input[:, t], hidden_ab[layer], A_list, resid_stats)

        # Decoder abnormal
        outputs_ab = []
        dec_input = go
        for t in range(self.args.horizon):
            for layer in range(self.num_layers):
                hidden_ab[layer] = self.decoder_ab[layer](dec_input, hidden_ab[layer], A_list, resid_stats)
            out = self.output_layer(hidden_ab[-1])
            outputs_ab.append(out)
            time_feat = x[:, -1, :, 1:] if y is None else y[:, t, :, 1:]
            dec_input = torch.cat([out, time_feat], dim=-1)

        # Step 4: Add normal + abnormal per timestep 
        outputs = [n + a for n, a in zip(outputs_normal, outputs_ab)]

        # Stack to [horizon, B, N, D]
        stacked = torch.stack(outputs, dim=0)  

        return stacked

    def extract_period(self, flow):
        # Stub: approximate previous day; in practice, precompute in dataloader if lag sufficient
        return flow[:, -self.args.lag:]  

    def extract_trend(self, flow):
        # Stub: approximate previous week
        return flow[:, -self.args.lag:]  

    def generate_distance_graph(self, x):
        dist = torch.cdist(self.node_embeddings, self.node_embeddings)
        Ad = torch.exp(- (dist ** 2) / 1.0)
        Ad = torch.where(Ad >= 0.1, Ad, torch.zeros_like(Ad))
        return Ad 

    def generate_latent_graph(self, x):
        E = self.node_embeddings
        curv = 1.0
        dist = torch.acosh(1 + curv * torch.cdist(E, E)**2 / ((1 - E.norm(dim=-1)**2).unsqueeze(1) * (1 - E.norm(dim=-1)**2).unsqueeze(0)))
        Al = torch.exp(-dist / 1.0)
        # Fixed: Use torch.where instead of inplace assignment
        Al = torch.where(Al >= 0.1, Al, torch.zeros_like(Al))
        return Al  

    def generate_grid_graph(self, grid_features):
        E = torch.cat(grid_features, dim=-1)
        dist = torch.cdist(E, E)
        Asg = torch.exp(- (dist ** 2) / 1.0)
        values, indices = torch.topk(Asg, self.args.topk, dim=-1)
        # Fixed: Use functional torch.scatter instead of inplace scatter_
        Ag = torch.scatter(torch.zeros_like(Asg), -1, indices, values)
        return Ag

    def generate_abnormality_graph(self, x):
        resid = x[..., 0] - x[..., 0].mean(dim=1, keepdim=True) 
        U = resid.permute(0, 2, 1) 
        corr = F.cosine_similarity(U.unsqueeze(1), U.unsqueeze(2), dim=-1) 
        Adelta = (corr > 0.5).float()
        resid_stats = torch.stack([resid.mean(dim=1), resid.var(dim=1)], dim=-1) 
        return Adelta, resid_stats

    def region_divide(self, A):
        if A.dim() == 2:
            N = A.size(0)
            is_static = True
        else:
            B, N, _ = A.shape
            is_static = False
        reg_sizes = [N // self.args.regions] * self.args.regions
        extra = N % self.args.regions
        for i in range(extra):
            reg_sizes[i] += 1
        starts = [0]
        for s in reg_sizes:
            starts.append(starts[-1] + s)
        if is_static:
            return [A[starts[i]:starts[i+1], starts[i]:starts[i+1]].clone() for i in range(self.args.regions)] 
        else:
            return [A[:, starts[i]:starts[i+1], starts[i]:starts[i+1]].clone() for i in range(self.args.regions)] 


class DMGCGRUCell(nn.Module):
    def __init__(self, rnn_units, input_dim, args):
        super(DMGCGRUCell, self).__init__()
        self.rnn_units = rnn_units
        self.g_u = DMGCBlock(input_dim + rnn_units, rnn_units, args)
        self.g_r = DMGCBlock(input_dim + rnn_units, rnn_units, args)
        self.g_c = DMGCBlock(input_dim + rnn_units, rnn_units, args)

    def forward(self, x_t, h_prev, A_list, resid_stats):
        inp = torch.cat([x_t, h_prev], dim=-1)
        U, _ = self.g_u(inp, A_list, resid_stats)
        R, _ = self.g_r(inp, A_list, resid_stats)
        cand_inp = torch.cat([x_t, R * h_prev], dim=-1)
        Ht, _ = self.g_c(cand_inp, A_list, resid_stats)
        u = torch.sigmoid(U)
        h_tilde = torch.tanh(Ht)
        h = u * h_tilde + (1 - u) * h_prev
        return h

class DMGCBlock(nn.Module):
    def __init__(self, d_in, d_out, args):
        super(DMGCBlock, self).__init__()
        self.graphs = 4  # Ad, Al, Ag, Adelta
        self.regions = args.regions
        self.W = nn.ParameterList([nn.Parameter(torch.randn(d_in, d_out)) for _ in range(self.graphs)])
        self.att1 = nn.Linear(self.graphs * d_out + 2, d_out)  # Adjusted: removed * self.regions since we cat after reconstruction
        self.att2 = nn.Linear(d_out, self.graphs)

    def forward(self, h, A_list, resid_stats):
        B, N, _ = h.shape

        # Compute region starts and ends (handles uneven division)
        reg_sizes = [N // self.regions] * self.regions
        extra = N % self.regions
        for i in range(extra):
            reg_sizes[i] += 1
        starts = [0]
        for s in reg_sizes:
            starts.append(starts[-1] + s)

        outs = []
        for k in range(self.graphs):
            reg_outs = []
            for r in range(self.regions):
                s, e = starts[r], starts[r + 1]
                h_reg = h[:, s:e, :]
                A_reg_r = A_list[k][r]  
                if A_reg_r.dim() == 2:
                    A_reg_r = A_reg_r.unsqueeze(0).expand(B, -1, -1)
                D_r = A_reg_r.sum(-1).clamp(min=1e-5).pow(-0.5)  
                size_r = e - s
                eye = torch.eye(size_r, device=h.device).unsqueeze(0).expand(B, -1, -1)
                A_norm = D_r.unsqueeze(2) * (A_reg_r + eye) * D_r.unsqueeze(1)  
                h_w = h_reg @ self.W[k]  
                reg_out = A_norm @ h_w  
                reg_outs.append(F.relu(reg_out))
            Hk = torch.cat(reg_outs, dim=1)  
            outs.append(Hk)
        H = torch.cat(outs, dim=-1)  

        # Attention with resid_stats
        resid_stats_exp = resid_stats  
        z_inp = torch.cat([H, resid_stats_exp], dim=-1)  
        z = F.relu(self.att1(z_inp))
        alpha = F.softmax(self.att2(z), dim=-1)  

        # Bias alpha based on decomposition (new: higher for Adelta if high resid var)
        bias = (resid_stats[:, :, 1] > 0.5).float()  
        bias_mat = bias.unsqueeze(-1).expand(-1, -1, self.graphs)  
        bias_weights = torch.tensor([0.1, 0.1, 0.1, 1.0], device=h.device).reshape(1, 1, self.graphs).expand(B, N, -1)
        bias = bias_mat * bias_weights
        alpha = alpha * (1 + bias)
        alpha = alpha / alpha.sum(-1, keepdim=True)

        # Fuse per graph
        O = sum(alpha[:, :, i].unsqueeze(-1) * outs[i] for i in range(self.graphs))  
        return O, alpha

# New: Dynamic CPT Fusion
class CPTFusion(nn.Module):
    def __init__(self, d):
        super(CPTFusion, self).__init__()
        self.fc1 = nn.Linear(3 * d, d) 
        self.fc2 = nn.Linear(d, 3)

    def forward(self, Mc, Mp, Mt):
        X = torch.cat([Mc, Mp, Mt], dim=-1)  
        z = F.relu(self.fc1(X))
        alpha = F.softmax(self.fc2(z), dim=-1)  
        Xf = alpha[..., 0].unsqueeze(-1) * Mc + alpha[..., 1].unsqueeze(-1) * Mp + alpha[..., 2].unsqueeze(-1) * Mt
        return torch.tanh(Xf)


### basic Trainer ###
import torch
import os
import time
from lib.metrics import MAE_torch, MSE_torch, RMSE_torch
from lib.logger import get_logger 

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
        if val_loader is not None:
            self.val_per_epoch = len(val_loader)
        self.logger = get_logger(self.args.log_dir, name="trainer", debug=True)
        self.logger.info("Trainer initialized successfully.")
        self.logger.info(f"Training batches per epoch: {self.train_per_epoch}")
        self.logger.info(f"Validation batches per epoch: {self.val_per_epoch if val_loader else 'None'}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.logger.info(f"Starting training for epoch {epoch}...")
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.args.device)
            target = target.to(self.args.device)
            target_flow = target[..., :self.args.output_dim]  
            self.optimizer.zero_grad()
            output = self.model(data, target) 
            output = output.permute(1, 0, 2, 3)  
            if self.args.real_value:
                output = self.scaler.inverse_transform(output)
            loss_value = self.loss(output, target_flow)
            loss_value.backward()
            self.optimizer.step()
            total_loss += loss_value.item()
            # Log every 100 batches to avoid flooding
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == self.train_per_epoch:
                elapsed = time.time() - start_time
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}/{self.train_per_epoch}, Batch Loss: {loss_value.item():.4f}, Time elapsed: {elapsed:.2f}s")
                print(f"Epoch {epoch}, Batch {batch_idx + 1}/{self.train_per_epoch}, Batch Loss: {loss_value.item():.4f}, Time elapsed: {elapsed:.2f}s")  # Also print to console
        avg_loss = total_loss / self.train_per_epoch
        self.logger.info(f"Epoch {epoch} completed. Average Train Loss: {avg_loss:.4f}")
        return avg_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        self.logger.info(f"Starting validation for epoch {epoch}...")
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                target_flow = target[..., :self.args.output_dim]
                output = self.model(data, target)
                output = output.permute(1, 0, 2, 3)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                loss_value = self.loss(output, target_flow)
                total_val_loss += loss_value.item()
                # Log every 100 batches
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == self.val_per_epoch:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Epoch {epoch}, Val Batch {batch_idx + 1}/{self.val_per_epoch}, Batch Loss: {loss_value.item():.4f}, Time elapsed: {elapsed:.2f}s")
                    print(f"Epoch {epoch}, Val Batch {batch_idx + 1}/{self.val_per_epoch}, Batch Loss: {loss_value.item():.4f}, Time elapsed: {elapsed:.2f}s")
        avg_val_loss = total_val_loss / self.val_per_epoch
        self.logger.info(f"Validation for epoch {epoch} completed. Average Val Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def train(self):
        min_val_loss = float('inf')
        self.logger.info(f"Starting training for {self.args.epochs} epochs...")
        for epoch in range(self.args.epochs):
            self.logger.info(f"--- Epoch {epoch}/{self.args.epochs - 1} ---")
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), self.args.log_dir + "best_model.pth")
                self.logger.info(f"New best model saved with Val Loss: {val_loss:.4f}")
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            self.logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def MAPE_torch(self, pred, true, mask_value=None):
        if mask_value != None:
            mask = torch.gt(true, mask_value)
            pred = torch.masked_select(pred, mask)
            true = torch.masked_select(true, mask)
        return torch.mean(torch.abs(torch.div((true - pred), (true+0.001))))
    
    def MAE_torch(self, pred, true, mask_value=None):
        if mask_value != None:
            mask = torch.gt(true, mask_value)
            pred = torch.masked_select(pred, mask)
            true = torch.masked_select(true, mask)
        return torch.mean(torch.abs(true-pred))

    def MSE_torch(self, pred, true, mask_value=None):
        if mask_value != None:
            mask = torch.gt(true, mask_value)
            pred = torch.masked_select(pred, mask)
            true = torch.masked_select(true, mask)
        return torch.mean((pred - true) ** 2)

    def RMSE_torch(self, pred, true, mask_value=None):
        if mask_value != None:
            mask = torch.gt(true, mask_value)
            pred = torch.masked_select(pred, mask)
            true = torch.masked_select(true, mask)
        return torch.sqrt(torch.mean((pred - true) ** 2))

    def test(self, model, args, test_loader, scaler, logger=None):
        model.eval()
        with torch.no_grad():
            mae, mse, rmse, mape = 0, 0, 0, 0
            self.logger.info("Starting testing...")
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(args.device)
                target = target.to(args.device)
                target_clone = target.clone()
                target_clone[..., 0] = 0  # Zero out flow to prevent potential info leak (though not used)
                target_flow = target[..., :args.output_dim]
                output = model(data, target_clone)
                output = output.permute(1, 0, 2, 3)
                if args.real_value:
                    output = scaler.inverse_transform(output)
                mae += self.MAE_torch(output, target_flow, 0.0).item()
                rmse += self.RMSE_torch(output, target_flow).item()
                mape += self.MAPE_torch(output, target_flow, 0.0).item()
                # Optional: Log per batch during test if needed
                if (batch_idx + 1) % 100 == 0:
                    self.logger.info(f"Test Batch {batch_idx + 1}/{len(test_loader)}, MAE: {mae / (batch_idx + 1):.4f}")
            avg_mae = mae / len(test_loader)
            avg_rmse = rmse / len(test_loader)
            avg_mape = mape / len(test_loader)
            print(f"Test: MAE {avg_mae:.4f}, RMSE {avg_rmse:.4f}, MAPE {avg_mape:.4f}")
            self.logger.info(f"Test completed: MAE {avg_mae:.4f}, RMSE {avg_rmse:.4f}, MAPE {avg_mape:.4f}")