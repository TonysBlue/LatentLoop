from __future__ import annotations

import torch
from torch import Tensor, nn

from latentloop.action_tokens import ActionToken, ActionTokenizer
from latentloop.config import ModelConfig
from latentloop.types import ActionLocalState


class ActionHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.burst_tokens = config.action_burst_tokens
        self.tokenizer = ActionTokenizer(config.max_action_duration_ms, self.burst_tokens)
        self.token_embedding = nn.Embedding(self.tokenizer.vocab_size, config.model_dim)
        self.context = nn.Linear(config.model_dim, config.model_dim)
        self.decoder = nn.GRUCell(config.model_dim, config.model_dim)
        self.output = nn.Linear(config.model_dim, self.tokenizer.vocab_size)

    def initial_state(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> ActionLocalState:
        return ActionLocalState(
            hidden=torch.zeros(batch, self.decoder.hidden_size, device=device, dtype=dtype),
            previous_token=torch.full(
                (batch,), int(ActionToken.PAD), device=device, dtype=torch.long
            ),
            active=torch.zeros(batch, device=device, dtype=torch.bool),
            event_type=torch.full((batch,), int(ActionToken.PAD), device=device, dtype=torch.long),
            burst_tokens=torch.zeros(batch, device=device, dtype=torch.long),
        )

    def forward(
        self,
        hidden: Tensor,
        state: ActionLocalState,
        teacher_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, ActionLocalState, Tensor]:
        context = self.context(hidden[:, -1])
        current = state.hidden
        previous = state.previous_token
        logits: list[Tensor] = []
        tokens: list[Tensor] = []
        masks: list[Tensor] = []
        active = state.active
        event_type = state.event_type
        burst_count = state.burst_tokens
        steps = self.burst_tokens
        ended = torch.zeros_like(active)
        for index in range(steps):
            candidate = self.decoder(self.token_embedding(previous) + context, current)
            current_logits = self.output(candidate)
            logits.append(current_logits)
            if teacher_tokens is not None:
                token = teacher_tokens[:, index]
                valid = teacher_tokens[:, index].ne(int(ActionToken.PAD))
            else:
                token = current_logits.argmax(dim=-1)
                valid = ~ended
            current = torch.where(valid[:, None], candidate, current)
            tokens.append(token)
            masks.append(valid)
            starting = (
                (~active)
                & valid
                & token.ne(int(ActionToken.END_ACTION))
                & token.ne(int(ActionToken.NOOP))
            )
            event_type = torch.where(starting, token, event_type)
            ended = ended | (valid & token.eq(int(ActionToken.END_ACTION)))
            active = torch.where(ended, torch.zeros_like(active), active | starting)
            burst_count = torch.where(active, burst_count + valid.long(), burst_count)
            previous = torch.where(valid, token, previous)
        output_logits = torch.stack(logits, dim=1)
        output_tokens = torch.stack(tokens, dim=1)
        output_mask = torch.stack(masks, dim=1)
        next_state = ActionLocalState(current, previous, active, event_type, burst_count)
        return output_logits, output_tokens, next_state, output_mask
