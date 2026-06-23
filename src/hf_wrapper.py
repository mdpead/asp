import os
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from src.model import Transformer


class EnCyConfig(PretrainedConfig):
    model_type = "en_cy_transformer"

    def __init__(
        self,
        vocab_size=16000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_enc_layers=6,
        num_dec_layers=6,
        max_length=256,
        dropout=0.1,
        pad_token_id=2,
        bos_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        **kwargs,
    ):
        kwargs.setdefault("is_encoder_decoder", True)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_hidden_layers = num_dec_layers
        self.max_length = max_length
        self.dropout = dropout


class _EncoderWrapper(torch.nn.Module):
    """Wrapper around transformer.encode() returning BaseModelOutput."""
    def __init__(self, transformer):
        super().__init__()
        self._transformer = transformer

    def forward(self, input_ids, attention_mask=None, **kwargs):
        mask = attention_mask.bool() if attention_mask is not None else None
        hidden = self._transformer.encode(input_ids, mask)
        return BaseModelOutput(last_hidden_state=hidden)


class EnCyForTranslation(PreTrainedModel, GenerationMixin):
    config_class = EnCyConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = Transformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_enc_layers=config.num_enc_layers,
            num_dec_layers=config.num_dec_layers,
            vocab_size=config.vocab_size,
            max_length=config.max_length,
            dropout=config.dropout,
        )
        self.post_init()

    def get_encoder(self):
        return _EncoderWrapper(self.transformer)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        labels=None,
        **kwargs,
    ):
        src_mask = attention_mask.bool() if attention_mask is not None else None
        if decoder_attention_mask is not None:
            tgt_mask = decoder_attention_mask.bool()
        else:
            tgt_mask = torch.ones(decoder_input_ids.shape, dtype=torch.bool, device=decoder_input_ids.device)

        if encoder_outputs is None:
            enc = self.transformer.encode(input_ids, src_mask)
        else:
            enc = encoder_outputs.last_hidden_state

        tgt_dec = self.transformer.decode(enc, decoder_input_ids, src_mask, tgt_mask)
        logits = self.transformer.output(tgt_dec)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=enc,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        encoder_outputs=None,
        **kwargs,
    ):
        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }

    @classmethod
    def from_run(cls, run_path):
        from src.utils import load_config
        config_dict = load_config(run_path)
        model_cfg = config_dict["model"]
        hf_config = EnCyConfig(
            vocab_size=config_dict["tokenizer"]["vocab_size"],
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"],
            d_ff=model_cfg["d_ff"],
            num_enc_layers=model_cfg["num_enc_layers"],
            num_dec_layers=model_cfg["num_dec_layers"],
            max_length=model_cfg["max_length"],
            dropout=model_cfg["dropout"],
        )
        model = cls(hf_config)
        checkpoints_dir = f"{run_path}/checkpoints"
        latest = max(int(f.split(".")[0]) for f in os.listdir(checkpoints_dir) if f.endswith(".pt"))
        checkpoint = torch.load(f"{checkpoints_dir}/{latest}.pt", map_location="cpu")
        model.transformer.load_state_dict(checkpoint["model_state_dict"])
        return model
