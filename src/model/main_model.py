import numpy as np
import torch
import torch.nn as nn
from src.model.diff_model import diff_CSDI
torch.set_printoptions(sci_mode=False)


class CSDI(nn.Module):
    """
    Conditional Score-based Diffusion model for Imputation with some modifications for our purpose.
    - https://arxiv.org/pdf/2107.03502
    """
    def __init__(self, config: dict, dataset_dim: int, device: torch.device):
        """
        Args:
            config (dict): configuration dictionary from 'config/train_config.yaml'
            dataset_dim (int): number of features in the dataset
            device (torch.device): device to host the model
        """
        super().__init__()
        self.device = device

        # number of features in the dataset
        self.dataset_dim = dataset_dim

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


    def process_data(self, observed_data: torch.Tensor, presence_mask: torch.Tensor, feature_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders tensor dimensions, converts to float, and moves to device. Also creates the observed time points tensor.

        Args:
            observed_data (torch.Tensor): data tensor
            presence_mask (torch.Tensor): mask indicating presence of data for training
            feature_id (torch.Tensor): tensor indicating the id of each feature

        Returns:
            tuple: processed observed data, presence mask, observed time points, and feature id tensors
        """

        B, L, K = observed_data.shape

        observed_data = observed_data.to(self.device).float()
        presence_mask = presence_mask.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        presence_mask = presence_mask.permute(0, 2, 1)

        observed_tp = torch.arange(L).expand(B, -1).to(self.device) * 1.0
        feature_id = feature_id.to(self.device)
        
        return (
            observed_data,
            presence_mask,            
            observed_tp,
            feature_id, 
        )

    
    def time_embedding(self, pos: torch.Tensor, d_model: int=128) -> torch.Tensor:
        """
        Positional encoding for time embedding.

        Args:
            pos (torch.Tensor): tensor of observed time points
            d_model (int): time embedding dimension

        Returns:
            torch.Tensor: positional encoding tensor
        """

        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


    def get_side_info(self, observed_tp: torch.Tensor, presence_mask: torch.Tensor, feature_id: torch.Tensor) -> torch.Tensor:
        """
        Gets time and feature embeddings to use as side information for the diffusion model.
        The time embedding is a positional encoding of the observed time points, and the feature embedding is an embedding of the feature ids.
        The presence mask is also included in the side information.

        Args:
            observed_tp (torch.Tensor): tensor of observed time points
            presence_mask (torch.Tensor): mask indicating presence of data for training
            feature_id (torch.Tensor): tensor indicating the id of each feature

        Returns:
            torch.Tensor: concatenated side information tensor
        """
        B, K, L = presence_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B,L,K,emb)

        feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
            
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        side_mask = presence_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1) # (B,*+1,K,L)

        return side_info


    def calc_loss_valid(
        self, observed_data: torch.Tensor, presence_mask: torch.Tensor, side_info: torch.Tensor, is_train: bool
    ) -> torch.Tensor:
        """
        Calculates the validation loss averaged over diffusion steps. 

        Args:
            observed_data (torch.Tensor): data tensor
            presence_mask (torch.Tensor): mask indicating presence of data for training
            side_info (torch.Tensor): side information tensor
            is_train (bool): whether the model is in training mode

        Returns:
            torch.Tensor: average loss over diffusion steps
        """
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, presence_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps


    def calc_loss(
        self, observed_data: torch.Tensor, presence_mask: torch.Tensor, side_info: torch.Tensor, is_train: bool, set_t: int=-1
    ) -> torch.Tensor:
        """
        Performs a forward pass for a single random diffusion step and calculates the loss.
        
        Args:
            observed_data (torch.Tensor): data tensor
            presence_mask (torch.Tensor): mask indicating presence of data for training
            side_info (torch.Tensor): side information tensor
            is_train (bool): whether the model is in training mode
            set_t (int): diffusion step to calculate loss for

        Returns:
            torch.Tensor: loss for the diffusion step
        """
        
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


    def set_input_to_diffmodel(self, noisy_data: torch.Tensor, observed_data: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        """
        Concatenates the observed data and noisy data to create the input for the diffusion model.
        The observed data is masked by the presence mask, and the noisy data is masked by the inverse of the presence mask.

        Args:
            noisy_data (torch.Tensor): noisy data tensor
            observed_data (torch.Tensor): observed data tensor
            presence_mask (torch.Tensor): mask indicating presence of data for training
        
        Returns:
            torch.Tensor: concatenated input tensor for the diffusion model
        """
        cond_obs = (presence_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - presence_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input


    def impute(
        self,
        observed_data: torch.Tensor,
        presence_mask: torch.Tensor,
        side_info: torch.Tensor,
        n_samples: int,
        gen_noise_magnitude: float=1.0
        ):
        """
        Performs diffusion denoising to impute missing data in observed_data.

        Args:
            observed_data (torch.Tensor): data tensor
            presence_mask (torch.Tensor): mask indicating presence of data for training
            side_info (torch.Tensor): side information tensor
            n_samples (int): number of samples to generate
            gen_noise_magnitude (float): magnitude of noise to add during generation. 1.0 corresponds to the original noise level 
                added during the forward process.

        Returns:
            torch.Tensor: imputed data tensor
        """
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
                noise = torch.randn_like(current_sample) * gen_noise_magnitude
                sigma = (
                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                ) ** 0.5
                current_sample += sigma * noise

        imputed_samples = current_sample * (1 - presence_mask) + observed_data * presence_mask

        return imputed_samples


    def forward(self, observed_data: torch.Tensor, presence_mask: torch.Tensor, feature_id: torch.Tensor, is_train: int=1) -> torch.Tensor:
        """
        Performs a forward pass through the model and calculates the loss. If is_train is 0, it performs multiple forward passes and calculates the average loss.

        Args:
            observed_data (torch.Tensor): data tensor
            presence_mask (torch.Tensor): mask indicating presence of data for training
            feature_id (torch.Tensor): tensor indicating the id of each feature
            is_train (int): whether the model is in training mode

        Returns:
            torch.Tensor: loss tensor
        """
        (
            observed_data, # [B, K, L]
            presence_mask, # [B, K, L]            
            observed_tp, # [B, L]
            feature_id, # [B, K]
        ) = self.process_data(observed_data, presence_mask, feature_id)
        
        side_info = self.get_side_info(observed_tp, presence_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, presence_mask, side_info, is_train)


    def generate(
        self,
        observed_data: torch.Tensor,
        presence_mask: torch.Tensor,
        feature_id: torch.Tensor,
        n_samples: int=1,
        gen_noise_magnitude: float=1
        ) -> torch.Tensor:
        """
        Imputes missing data in observed_data using the diffusion model. The model generates n_samples of imputed data.

        Args:
            observed_data (torch.Tensor): data tensor
            presence_mask (torch.Tensor): mask indicating presence of data for training
            feature_id (torch.Tensor): tensor indicating the id of each feature
            n_samples (int): number of samples to generate
            gen_noise_magnitude (float): magnitude of noise to add during generation. 1.0 corresponds to the original noise level 
                added during the forward process.

        Returns:
            torch.Tensor: imputed data tensor
        """
        (
            observed_data, # [B, K, L]
            presence_mask, # [B, K, L]
            observed_tp, # [B, L]
            feature_id, # [B, K]
        ) = self.process_data(observed_data, presence_mask, feature_id)

        with torch.no_grad():
            side_info = self.get_side_info(observed_tp, presence_mask, feature_id)
            samples = self.impute(observed_data, presence_mask, side_info, n_samples, gen_noise_magnitude)
        
        return samples.permute(0, 2, 1) # [n_samples, L, K]