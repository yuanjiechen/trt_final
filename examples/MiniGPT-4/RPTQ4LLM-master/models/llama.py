import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb

class LLAMClass(BaseLM):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self._device = torch.device("cuda:0")
        self.model_name = args.model
        self.model = LlamaForCausalLM.from_pretrained(args.cache_dir, torch_dtype="auto")
        # self.model.generate()
        self.model.eval()
        self.seqlen = 128#self.model.config.max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir=args.cache_dir, use_fast=False, pretrained_model_name_or_path=args.cache_dir)
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings
    
    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256


    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return 1  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        pdb.set_trace()
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
LLAMA = LLAMClass