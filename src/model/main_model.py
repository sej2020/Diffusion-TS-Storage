import numpy as np
import torch
import torch.nn as nn
from src.model.diff_models import diff_CSDI
torch.set_printoptions(sci_mode=False)


class CSDI(nn.Module):
    def __init__(self, config, dataset_dim, device):
        super().__init__()
        self.device = device

        # number of features in the dataset
        self.dataset_dim = dataset_dim

        self.training_feature_sample_size = config["model"]["training_feature_sample_size"]

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.dataset_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        self.diffmodel = diff_CSDI(config_diff)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)


    def process_data(self, observed_data, presence_mask, feature_id=None):

        B, L, K = observed_data.shape

        observed_data = observed_data.to(self.device).float()
        presence_mask = presence_mask.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        presence_mask = presence_mask.permute(0, 2, 1)

        observed_tp = torch.arange(L).expand(B, -1).to(self.device) * 1.0
        if feature_id is None:
            feature_id=torch.arange(K).unsqueeze(0).expand(B,-1).to(self.device)
        else:
            feature_id = feature_id.to(self.device)
        
        return (
            observed_data,
            observed_tp,
            presence_mask,
            feature_id, 
        )


    def sample_features(self, observed_data, presence_mask, feature_id):
        size = self.training_feature_sample_size
        extracted_data = []
        extracted_presence_mask = []
        extracted_feature_id = []

        for k in range(len(observed_data)):
            ind = np.random.choice(self.dataset_dim, self.dataset_dim, replace=False) # random permutation
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_presence_mask.append(presence_mask[k,ind[:size]])   
            extracted_feature_id.append(feature_id[k,ind[:size]])
        
        extracted_data = torch.stack(extracted_data,0)
        extracted_presence_mask = torch.stack(extracted_presence_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        return extracted_data, extracted_presence_mask, extracted_feature_id


    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


    def get_side_info(self, observed_tp, presence_mask, feature_id=None):
        B, K, L = presence_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B,L,K,emb)

        if feature_id is not None:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        else:
            feature_embed = self.embed_layer(
                torch.arange(K).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B,L,-1,-1)
            
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        side_mask = presence_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1) # (B,*+1,K,L)

        return side_info


    def calc_loss_valid(
        self, observed_data, presence_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, presence_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps


    def calc_loss(
        self, observed_data, presence_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, presence_mask)
        # predicts the noise tensor that was added to observed data to yield noisy data
        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        target_mask = 1 - presence_mask # 1s where we would like to predict

        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss


    def set_input_to_diffmodel(self, noisy_data, observed_data, presence_mask):
        cond_obs = (presence_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - presence_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input


    def impute(self, observed_data, presence_mask, side_info, n_samples, generation_variance):
        B, K, L = observed_data.shape

        if B == 1: # B will be 1 all of the time I think
            observed_data = observed_data.expand(n_samples, -1, -1)
            presence_mask = presence_mask.expand(n_samples, -1, -1)
            side_info = side_info.expand(n_samples, -1, -1, -1)
        else:
            raise ValueError("Batch size should be set to 1 for generation.")

        current_sample = torch.randn_like(observed_data)

        for t in range(self.num_steps - 1, -1, -1):
            cond_obs = (presence_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - presence_mask) * current_sample).unsqueeze(1)
            diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

            predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * predicted)

            if t > 0:
                noise = torch.randn_like(current_sample) * generation_variance
                sigma = (
                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                ) ** 0.5
                current_sample += sigma * noise

        imputed_samples = current_sample * (1 - presence_mask) + observed_data * presence_mask
        return imputed_samples


    def forward(self, observed_data, presence_mask, is_train=1):
        (
            observed_data, # [B, K, L]
            observed_tp, # [B, L]
            presence_mask, # [B, K, L]
            feature_id, 
        ) = self.process_data(observed_data, presence_mask)
        
        if is_train == 1 and (self.dataset_dim > self.training_feature_sample_size):
            observed_data, presence_mask, feature_id = self.sample_features(observed_data, presence_mask, feature_id)
        else:
            feature_id = None
        
        side_info = self.get_side_info(observed_tp, presence_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, presence_mask, side_info, is_train)


    def generate(self, observed_data, presence_mask, feature_id, n_samples, generation_variance):
        (
            observed_data, # [B, K, L]
            observed_tp, # [B, L]
            presence_mask, # [B, K, L]
            feature_id, # [B, K]
        ) = self.process_data(observed_data, presence_mask, feature_id)

        with torch.no_grad():
            side_info = self.get_side_info(observed_tp, presence_mask, feature_id)

            print('Imputing', flush=True)
            samples = self.impute(observed_data, presence_mask, side_info, n_samples, generation_variance)
        
        return samples.permute(0, 2, 1) # [n_samples, L, K]