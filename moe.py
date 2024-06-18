import torch
import torch.nn as nn
import torch.nn.functional as F



class Expert(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x) -> torch.Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))



class NoisyTopkRouter(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.topkroute_linear = nn.Linear(args.dim, args.num_experts)
        self.noise_linear = nn.Linear(args.dim, args.num_experts)

    
    def forward(self, x: torch.Tensor):
        # x is the output tensor from multihead self attention block
        logits = self.topkroute_linear(x)

        #Noise logits
        noise_logits = self.noise_linear(x)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.args.num_experts_per_tok, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices



class SparseMoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.router = NoisyTopkRouter(args)
        self.experts = nn.ModuleList([Expert(args) for _ in range(args.num_experts)])

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output