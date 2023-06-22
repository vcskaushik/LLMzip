# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
import numpy as np
import zlib
import sys
import binascii


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
    
    def generate2(
        self,
        prompts: List[str],
        max_gen_len: int=256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded,t

    def encode(
        self,
        prompt: str,
        max_gen_len: int = 0,
        quantize: bool = False
    ):
        
        bsz = 1    # no of prompts = 1
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        prompt_size = len(prompt_tokens)

        total_len = min(params.max_seq_len, max_gen_len + prompt_size)
        
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
        print("True Tokens",tokens)
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = 1
        prev_pos = 0
        ranks_np = torch.clone(tokens).detach()
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            
            probs = torch.softmax(logits, dim=-1)
            if cur_pos > 50 :
                rank = gen_rank(probs,next_token=tokens[:,cur_pos],embeddings_fn =self.model.tok_embeddings, quantize_sim=True)
            else:
                rank = gen_rank(probs,next_token=tokens[:,cur_pos])
            #curr_rank_emb = self.model.tok_embeddings(tokens[:,cur_pos:cur_pos+1])
            #print("Embeddings",curr_rank_emb.shape,less_rank_idx.shape,tokens[:,cur_pos:cur_pos+1].shape)
            
            prev_pos = cur_pos
            
            ranks_np[:,cur_pos] = rank
        
        ranks_np2 = ranks_np.cpu().numpy()
        ranks_dtype = ranks_np2.dtype
        ranks_uncompressed = bytes(np.array2string(ranks_np2.reshape((-1,))[1:-1]), 'ascii')
        ranks_compressed = zlib.compress(ranks_np2, -1)
        true_tokens = tokens.cpu().numpy()

        if quantize == True:
            #ranks_np2_q =(np.rint(ranks_np2/5)*5).astype(int)
            ranks_np2_q = ranks_np2-1
            ranks_np2_q[ranks_np2_q==-1]=0
            ranks_dtype_q = ranks_np2_q.dtype
            ranks_compressed_q = zlib.compress(ranks_np2_q, -1)
            ranks_uncompressed_q = bytes(np.array2string(ranks_np2_q.reshape((-1,))[1:-1]), 'ascii')

        print("True Ranks",ranks_np2,ranks_np2.shape,ranks_dtype)


        
        string_bytes = bytes(prompt,'ascii')
        string_bytes_compressed = zlib.compress(string_bytes, -1)

        
        if quantize:
            print("Original String Size: ",len(string_bytes),"Original String Compressed Size: ",len(string_bytes_compressed)," Original Rank Size: ",len(ranks_uncompressed)," Compressed Rank Size: ", len(ranks_compressed),"Original Quantized Rank Size: ",len(ranks_uncompressed_q)," Compressed Quantized Rank Size: ", len(ranks_compressed_q))
            ranks_out_list = [ranks_compressed,ranks_dtype,ranks_np2,ranks_compressed_q,ranks_dtype_q,ranks_np2_q]
        else :
            print("Original String Size: ",len(string_bytes),"Original String Compressed Size: ",len(string_bytes_compressed)," Original Rank Size: ",len(ranks_uncompressed)," Compressed Rank Size: ", len(ranks_compressed))
            ranks_out_list = [ranks_compressed,ranks_dtype,ranks_np2]


        
        return ranks_out_list,true_tokens
            
    def decode(
        self,
        ranks_compressed,
        ranks_dtype,
        max_gen_len: int = 0,
        best: bool = False
    ):
        ranks = np.frombuffer(zlib.decompress(ranks_compressed),ranks_dtype)
        
        bsz = 1    # no of prompts = 1
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        total_len = ranks.shape[0]
        
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).long()
        tokens[:,0] = torch.tensor(ranks[0]).long()
        tokens = tokens.cuda()
        if best:
            best_tokens = tokens.clone().detach().cuda()
        ranks = torch.tensor(ranks).reshape(bsz,-1).cuda()
        
        start_pos = 1
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            probs = torch.softmax(logits, dim=-1)
            next_token = gen_next_token(probs,ranks[:,cur_pos:cur_pos+1])
            tokens[:,cur_pos] = torch.tensor(next_token).long()
            if best:
                best_tokens[:,cur_pos] = torch.tensor(torch.argmax(logits, dim=-1)).long()
            prev_pos = cur_pos
            
        decoded = []
        if best:
            decoded_best = []
            best_tokens_list = best_tokens.tolist()
        for i, t in enumerate(tokens.tolist()):
#             # cut to max gen len
#             t = t[: len(prompt_tokens[i]) + max_gen_len]
#             # cut to eos tok if any
#             try:
#                 t = t[: t.index(self.tokenizer.eos_id)]
#             except ValueError:
#                 pass
            decoded.append(self.tokenizer.decode(t))
            if best:
                decoded_best.append(self.tokenizer.decode(best_tokens_list[i]))
            
        tokens_out = tokens.cpu().numpy() 
        if best:
            best_tokens_out = best_tokens.cpu().numpy()
            return decoded[0],tokens_out,decoded_best[0],best_tokens_out
        else:
            return decoded[0],tokens_out
    
    def sim_compression(
        self,
        prompts: List[str],
        max_gen_len: int = 0,
        quantize: bool = False,
        best: bool = False
    ):
        true_tokens_list =[]
        decoded_list = []
        decoded_best_list = []
        decoded_list_quantized = []
        ranks_list = []
        ranks_list_quantized = []
        decoded_tokens_list = []
        decoded_best_tokens_list = []
        decoded_tokens_list_quantized = []
        for prompt in prompts:
            print("Input : ",prompt,"\n")
            ranks_out_list,true_tokens = self.encode(prompt,max_gen_len,quantize)
            true_tokens_list += [true_tokens]
            print("True tokens : ", true_tokens,"\n")
            
            # 0: ranks_compressed, 1: ranks_dtype,2: ranks_np2, 3: ranks_compressed_q, 4:ranks_dtype_q, 5:ranks_np2_q
            ranks_list += [ranks_out_list[2]]
            print("Ranks : ", ranks_list[-1])
            if best:
                decoded , decoded_tokens, decoded_best,decoded_best_tokens = self.decode(ranks_out_list[0],ranks_out_list[1],max_gen_len,best)
                decoded_list += [decoded]
                decoded_tokens_list += [decoded_tokens]
                decoded_best_list += [decoded_best]
                decoded_best_tokens_list += [decoded_best_tokens]
                print("Decoded : ",decoded,"\n")
                print("Decoded Tokens : ",decoded_tokens,"\n")
                print("Decoded Best : ",decoded_best,"\n")
                print("Decoded Best Tokens : ",decoded_best_tokens,"\n")
            else:
                decoded , decoded_tokens = self.decode(ranks_out_list[0],ranks_out_list[1],max_gen_len)
                decoded_list += [decoded]
                decoded_tokens_list += [decoded_tokens]
                print("Decoded : ",decoded,"\n")
                print("Decoded Tokens : ",decoded_tokens,"\n")

            if quantize:    
                ranks_list_quantized += [ranks_out_list[5]]
                print("Ranks Quantized : ", ranks_list_quantized[-1],"\n")
                decoded_q , decoded_tokens_q = self.decode(ranks_out_list[3],ranks_out_list[4],max_gen_len)
                decoded_list_quantized += [decoded_q]
                decoded_tokens_list_quantized += [decoded_tokens_q]
                print("Decoded Quantized : ",decoded_q,"\n")
                print("Decoded Tokens Quantized : ",decoded_tokens_q,"\n")
           
        print("*********************************************")      
                
        if best:
            if quantize:
                return true_tokens_list, decoded_list, ranks_list, decoded_tokens_list,decoded_best_list,decoded_best_tokens_list, decoded_list_quantized, ranks_list_quantized, decoded_tokens_list_quantized
            else:
                return true_tokens_list, decoded_list, ranks_list, decoded_tokens_list,decoded_best_list, decoded_best_tokens_list
        else:
            if quantize:
                return true_tokens_list, decoded_list, ranks_list, decoded_tokens_list, decoded_list_quantized, ranks_list_quantized, decoded_tokens_list_quantized
            else:
                return true_tokens_list, decoded_list, ranks_list, decoded_tokens_list
    
    def sim_compression(
        self,
        prompts: List[str],
        max_gen_len: int = 0,
        quantize: bool = False,
        best: bool = False
    ):
        true_tokens_list =[]
        decoded_list = []
        decoded_best_list = []
        decoded_list_quantized = []
        ranks_list = []
        ranks_list_quantized = []
        decoded_tokens_list = []
        decoded_best_tokens_list = []
        decoded_tokens_list_quantized = []
        for prompt in prompts:
            print("Input : ",prompt,"\n")
            ranks_out_list,true_tokens = self.encode(prompt,max_gen_len,quantize)
            true_tokens_list += [true_tokens]
            print("True tokens : ", true_tokens,"\n")
            
            # 0: ranks_compressed, 1: ranks_dtype,2: ranks_np2, 3: ranks_compressed_q, 4:ranks_dtype_q, 5:ranks_np2_q
            ranks_list += [ranks_out_list[2]]
            print("Ranks : ", ranks_list[-1])
            if best:
                decoded , decoded_tokens, decoded_best,decoded_best_tokens = self.decode(ranks_out_list[0],ranks_out_list[1],max_gen_len,best)
                decoded_list += [decoded]
                decoded_tokens_list += [decoded_tokens]
                decoded_best_list += [decoded_best]
                decoded_best_tokens_list += [decoded_best_tokens]
                print("Decoded : ",decoded,"\n")
                print("Decoded Tokens : ",decoded_tokens,"\n")
                print("Decoded Best : ",decoded_best,"\n")
                print("Decoded Best Tokens : ",decoded_best_tokens,"\n")
            else:
                decoded , decoded_tokens = self.decode(ranks_out_list[0],ranks_out_list[1],max_gen_len)
                decoded_list += [decoded]
                decoded_tokens_list += [decoded_tokens]
                print("Decoded : ",decoded,"\n")
                print("Decoded Tokens : ",decoded_tokens,"\n")

            if quantize:    
                ranks_list_quantized += [ranks_out_list[5]]
                print("Ranks Quantized : ", ranks_list_quantized[-1],"\n")
                decoded_q , decoded_tokens_q = self.decode(ranks_out_list[3],ranks_out_list[4],max_gen_len)
                corrected_in = ["Can you retrieve this corrupted english text : " + decoded_q]
                decoded_q2 , decoded_tokens_q2 = self.generate2(corrected_in)
                decoded_list_quantized += [decoded_q]
                decoded_tokens_list_quantized += [decoded_tokens_q]
                print("Decoded Quantized : ",decoded_q,"\n")
                print("Decoded Tokens Quantized : ",decoded_tokens_q,"\n")
                print("Decoded Quantized 2 : ",decoded_q2,"\n")
                print("Decoded Tokens Quantized 2 : ",decoded_tokens_q2,"\n")
           
        print("*********************************************")      
                
        if best:
            if quantize:
                return true_tokens_list, decoded_list, ranks_list, decoded_tokens_list,decoded_best_list,decoded_best_tokens_list, decoded_list_quantized, ranks_list_quantized, decoded_tokens_list_quantized
            else:
                return true_tokens_list, decoded_list, ranks_list, decoded_tokens_list,decoded_best_list, decoded_best_tokens_list
        else:
            if quantize:
                return true_tokens_list, decoded_list, ranks_list, decoded_tokens_list, decoded_list_quantized, ranks_list_quantized, decoded_tokens_list_quantized
            else:
                return true_tokens_list, decoded_list, ranks_list, decoded_tokens_list,

        


        



def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def gen_rank(probs,next_token,embeddings_fn =None,quantize_sim=False):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) 
    rank = torch.where(probs_idx == next_token)[-1]
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    with torch.no_grad():
        if (quantize_sim)and(rank>5):
            less_rank_idx = probs_idx[:,:rank]
            less_rank_emb = embeddings_fn(less_rank_idx)
            curr_rank_emb = embeddings_fn(probs_idx[:,rank:rank+1])
            sim_values = cos_sim(less_rank_emb,curr_rank_emb).squeeze()
            true_rank = rank.cpu().numpy()
            rank = torch.argmax(sim_values)
            print("Best Sim : ",torch.max(sim_values).cpu().numpy(),"rank selected",rank.cpu().numpy(),"True rank",true_rank)
    return rank

def gen_next_token(probs,rank):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    #print(probs_idx.shape,rank.shape)
    next_token = torch.gather(probs_idx, -1, rank)
    return next_token

