import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DDGCRN_PP(nn.Module):
    def __init__(self, args):
        super(DDGCRN_PP, self).__init__()
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
        # x: [B, lag, N, 3] (flow=0, day=1, week=2)
        B, T_lag, N, D = x.shape

        # Step 1: Extract CPT views from flow channel (fix: all from flow, time features used separately)
        flow = x[..., 0]  # [B, lag, N]
        closeness = flow[:, -self.args.lag:]  # Recent [B, lag, N]
        period = self.extract_period(flow)  # Implement proper extraction; stub: shift by steps_per_day
        trend = self.extract_trend(flow)  # Stub: shift by 7*steps_per_day
        views = [closeness, period, trend]

        # Process views to grid_features
        grid_features = []
        for v in views:
            pad = torch.zeros(B, T_lag, self.padded_nodes - N, device=x.device)
            v_padded = torch.cat([v, pad], dim=-1)  # [B, lag, padded]
            v_grid = v_padded.view(B, 1, T_lag, self.grid_h, self.grid_w)  # [B,1,lag,H,W]
            h = F.relu(self.conv3d(v_grid))  # [B,32,lag,H,W]
            r = self.res_unit(h) + h
            r_flat = r.view(B, 32, T_lag, -1)[:, :, :, :N]  # [B,32,lag,N]
            r_trans = r_flat.permute(0, 3, 1, 2).reshape(B, N, 32 * T_lag)  # [B,N,32*lag]
            grid_features.append(r_trans)

        # Step 2: Generate graphs
        Ad = self.generate_distance_graph(x)
        Al = self.generate_latent_graph(x)
        Ag = self.generate_grid_graph(grid_features)
        Adelta, resid_stats = self.generate_abnormality_graph(x)
        A_list = [self.region_divide(A) for A in [Ad, Al, Ag, Adelta]]

        # Step 3: Decomposition for normal and abnormal (per paper)
        flow_mean = flow.mean(dim=1, keepdim=True)  # [B,1,N]
        resid = flow - flow_mean  # [B,lag,N] approx abnormal (stub; paper uses reverse pred)
        repeated_mean = flow_mean.repeat(1, T_lag, 1).unsqueeze(-1)  # [B, lag, N, 1]
        time_feats = x[..., 1:]  # [B, lag, N, 2]
        normal_input = torch.cat([repeated_mean, time_feats], dim=-1)  # [B,lag,N,3]
        resid = resid.unsqueeze(-1)  # [B, lag, N, 1]
        abnormal_input = torch.cat([resid, time_feats], dim=-1)  # [B,lag,N,3]

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
            out = self.output_layer(hidden_normal[-1])  # [B, N, output_dim]
            outputs_normal.append(out)
            time_feat = x[:, -1, :, 1:] if y is None else y[:, t, :, 1:]  # [B, N, 2]
            dec_input = torch.cat([out, time_feat], dim=-1)  # [B, N, 3]; autoregressive

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

        # Step 4: Add normal + abnormal per timestep (per paper)
        outputs = [n + a for n, a in zip(outputs_normal, outputs_ab)]

        # Stack to [horizon, B, N, D]
        stacked = torch.stack(outputs, dim=0)  # [horizon, B, N, D]

        # Optional: Apply CPT fusion if intended (here skipped as decomposition is primary; can add if needed)
        return stacked

    def extract_period(self, flow):
        # Stub: approximate previous day; in practice, precompute in dataloader if lag sufficient
        return flow[:, -self.args.lag:]  # Placeholder

    def extract_trend(self, flow):
        # Stub: approximate previous week
        return flow[:, -self.args.lag:]  # Placeholder

    def generate_distance_graph(self, x):
        dist = torch.cdist(self.node_embeddings, self.node_embeddings)
        Ad = torch.exp(- (dist ** 2) / 1.0)
        # Fixed: Use torch.where instead of inplace assignment
        Ad = torch.where(Ad >= 0.1, Ad, torch.zeros_like(Ad))
        return Ad  # [N, N]

    def generate_latent_graph(self, x):
        E = self.node_embeddings
        curv = 1.0
        dist = torch.acosh(1 + curv * torch.cdist(E, E)**2 / ((1 - E.norm(dim=-1)**2).unsqueeze(1) * (1 - E.norm(dim=-1)**2).unsqueeze(0)))
        Al = torch.exp(-dist / 1.0)
        # Fixed: Use torch.where instead of inplace assignment
        Al = torch.where(Al >= 0.1, Al, torch.zeros_like(Al))
        return Al  # [N, N]

    def generate_grid_graph(self, grid_features):
        E = torch.cat(grid_features, dim=-1)  # [B, N, 3*32*lag]
        dist = torch.cdist(E, E)  # [B, N, N]
        Asg = torch.exp(- (dist ** 2) / 1.0)
        values, indices = torch.topk(Asg, self.args.topk, dim=-1)
        # Fixed: Use functional torch.scatter instead of inplace scatter_
        Ag = torch.scatter(torch.zeros_like(Asg), -1, indices, values)
        return Ag

    def generate_abnormality_graph(self, x):
        resid = x[..., 0] - x[..., 0].mean(dim=1, keepdim=True)  # [B, lag, N]
        U = resid.permute(0, 2, 1)  # [B, N, lag]
        corr = F.cosine_similarity(U.unsqueeze(1), U.unsqueeze(2), dim=-1)  # [B, N, N]
        Adelta = (corr > 0.5).float()
        resid_stats = torch.stack([resid.mean(dim=1), resid.var(dim=1)], dim=-1)  # [B, N, 2]
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
            return [A[starts[i]:starts[i+1], starts[i]:starts[i+1]].clone() for i in range(self.args.regions)]  # Added .clone() for safety
        else:
            return [A[:, starts[i]:starts[i+1], starts[i]:starts[i+1]].clone() for i in range(self.args.regions)]  # Added .clone() for safety


# New: Extended DMGC-GRU Cell (from 2112.02264v1 Fig 2, extended to 4 graphs)
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

# New: Region-aware DMGC Block (from 2112.02264v1 eq 6-7, extended with decomp-aware att)
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
                A_reg_r = A_list[k][r]  # [size_r, size_r] or [B, size_r, size_r]
                if A_reg_r.dim() == 2:
                    A_reg_r = A_reg_r.unsqueeze(0).expand(B, -1, -1)
                D_r = A_reg_r.sum(-1).clamp(min=1e-5).pow(-0.5)  # [B, size_r]
                size_r = e - s
                eye = torch.eye(size_r, device=h.device).unsqueeze(0).expand(B, -1, -1)
                A_norm = D_r.unsqueeze(2) * (A_reg_r + eye) * D_r.unsqueeze(1)  # [B, size_r, size_r]
                h_w = h_reg @ self.W[k]  # [B, size_r, d_out]
                reg_out = A_norm @ h_w  # [B, size_r, d_out]
                reg_outs.append(F.relu(reg_out))
            Hk = torch.cat(reg_outs, dim=1)  # [B, N, d_out]
            outs.append(Hk)
        H = torch.cat(outs, dim=-1)  # [B, N, graphs * d_out]

        # Attention with resid_stats
        resid_stats_exp = resid_stats  # [B, N, 2]
        z_inp = torch.cat([H, resid_stats_exp], dim=-1)  # [B, N, graphs * d_out + 2]
        z = F.relu(self.att1(z_inp))
        alpha = F.softmax(self.att2(z), dim=-1)  # [B, N, graphs]

        # Bias alpha based on decomposition (new: higher for Adelta if high resid var)
        bias = (resid_stats[:, :, 1] > 0.5).float()  # [B, N]
        bias_mat = bias.unsqueeze(-1).expand(-1, -1, self.graphs)  # [B, N, graphs]
        bias_weights = torch.tensor([0.1, 0.1, 0.1, 1.0], device=h.device).reshape(1, 1, self.graphs).expand(B, N, -1)
        bias = bias_mat * bias_weights
        alpha = alpha * (1 + bias)
        alpha = alpha / alpha.sum(-1, keepdim=True)

        # Fuse per graph
        O = sum(alpha[:, :, i].unsqueeze(-1) * outs[i] for i in range(self.graphs))  # [B, N, d_out]
        return O, alpha

# New: Dynamic CPT Fusion (improved over zhao2021.pdf eq 7-8)
class CPTFusion(nn.Module):
    def __init__(self, d):
        super(CPTFusion, self).__init__()
        self.fc1 = nn.Linear(3 * d, d)  # 3 views; add more for Zext if needed
        self.fc2 = nn.Linear(d, 3)

    def forward(self, Mc, Mp, Mt):
        X = torch.cat([Mc, Mp, Mt], dim=-1)  # [B, N, 3d]
        z = F.relu(self.fc1(X))
        alpha = F.softmax(self.fc2(z), dim=-1)  # [B, N, 3]; node-wise
        Xf = alpha[..., 0].unsqueeze(-1) * Mc + alpha[..., 1].unsqueeze(-1) * Mp + alpha[..., 2].unsqueeze(-1) * Mt
        return torch.tanh(Xf)


### basic Trainer ###
import torch
import os
import time
from lib.metrics import MAE_torch, MSE_torch, RMSE_torch
from lib.logger import get_logger  # Import the logger from your lib (assuming it's available)

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
        # Initialize logger (writes to args.log_dir/run.log)
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
            target_flow = target[..., :self.args.output_dim]  # [B, horizon, N, 1]
            self.optimizer.zero_grad()
            output = self.model(data, target)  # Now [horizon, B, N, 1]
            output = output.permute(1, 0, 2, 3)  # [B, horizon, N, 1]
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

    def test(self, model, args, test_loader, scaler, logger=None):
        model.eval()
        with torch.no_grad():
            mae, mse, rmse = 0, 0, 0
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
                mae += MAE_torch(output, target_flow, 0.0).item()
                mse += MSE_torch(output, target_flow).item()
                rmse += RMSE_torch(output, target_flow).item()
                # Optional: Log per batch during test if needed
                if (batch_idx + 1) % 100 == 0:
                    self.logger.info(f"Test Batch {batch_idx + 1}/{len(test_loader)}, MAE: {mae / (batch_idx + 1):.4f}")
            avg_mae = mae / len(test_loader)
            avg_mse = mse / len(test_loader)
            avg_rmse = rmse / len(test_loader)
            print(f"Test: MAE {avg_mae:.4f}, MSE {avg_mse:.4f}, RMSE {avg_rmse:.4f}")
            self.logger.info(f"Test completed: MAE {avg_mae:.4f}, MSE {avg_mse:.4f}, RMSE {avg_rmse:.4f}")
