'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    hie_allspark_worker.py
'''
from ..utils.hie_allspark.model_hie_allspark import (
    AllSparkM6Model,
)
from ..utils.hie_allspark.model_hie_allspark import (
    AllSparkRequest,
)
from dashinfer import allspark
import torch
from ..utils.mrope import build_mrope_positions


class HieAllsparkWorker:
    def __init__(self, as_model_config: allspark.AsModelConfig):
        self.model = AllSparkM6Model(as_model_config)

    def eval(self, request: AllSparkRequest):
        # qwenvl2 to get position list
        position_list = self.get_llm_positions(
            request.input_lists, request.vit_positions, request.vit_target_token
        )
        request.gen_cfg = self.get_gen_cfg(request, position_list)
        return self.model.forward(request)

    """
    compute llm positions with mrope
    """

    def get_llm_positions(
        self,
        total_input_ids,
        vit_grid_thw_list,
        image_modality_token_id,
    ):
        if len(total_input_ids) != len(vit_grid_thw_list):
            raise ValueError("token batches and vision grid batches must match")

        position_batches = []
        for input_tokens, vision_grids in zip(
            total_input_ids, vit_grid_thw_list
        ):
            if vision_grids and vision_grids[0] is None:
                return []
            position_batches.append(
                build_mrope_positions(
                    input_tokens,
                    vision_grids,
                    image_modality_token_id,
                )
            )

        sequence_lengths = {
            len(positions[0]) for positions in position_batches
        }
        if len(sequence_lengths) > 1:
            raise ValueError("M-RoPE batches must have equal sequence lengths")
        if not position_batches:
            return []
        return torch.tensor(position_batches, dtype=torch.int64).permute(
            1, 0, 2
        ).contiguous()

    def get_gen_cfg(self, request: AllSparkRequest, position_list: list):
        dl_list = []
        pos_dl_list = []
        # modify gen_cfg max_length
        # since truncate is done in 一体化, we need to modify max_length
        # logging.error(f"request.gen_cfg max_length:{request.gen_cfg['max_length']}, old_context_len:{request.old_context_len}, new_context_len:{request.new_context_len}")
        request.gen_cfg["max_length"] = (
            request.gen_cfg["max_length"]
            - request.old_context_len
            + request.new_context_len
        )
        if request.gen_cfg["max_length"] > request.max_total_tokens:
            request.gen_cfg["max_length"] = request.max_total_tokens
        if len(request.vit_embs) == 0:
            return request.gen_cfg
        for vit in request.vit_embs:
            dl_list.append(torch.utils.dlpack.to_dlpack(vit.to(torch.float32)))
        if len(position_list) > 0:
            pos_dl_list.append(
                torch.utils.dlpack.to_dlpack(position_list.to(torch.int32))
            )
        as_extra_embedding_info_0 = allspark.MultiMediaInfo()
        as_extra_embedding_info_0.set_multimedia_type(0)
        as_extra_embedding_info_0.add_multimedia_content(
            str(request.vit_target_token), dl_list
        )
        key_list = []
        if request.vit_keys is not None:
            for key in request.vit_keys:
                # logging.info(f"key shape: {key.shape}, key:{key}")
                key_list.append(torch.utils.dlpack.to_dlpack(key.to(torch.int32)))
            as_extra_embedding_info_0.add_multimedia_content("hash_input", key_list)
        if len(pos_dl_list) > 0:
            as_extra_embedding_info_0.add_multimedia_content("positions", pos_dl_list)
        request.gen_cfg["mm_info"] = as_extra_embedding_info_0
        request.gen_cfg["extra_embedding"] = dl_list
        request.gen_cfg["extra_embedding_pos"] = pos_dl_list
        request.gen_cfg["extra_embedding_key"] = key_list
        return request.gen_cfg

    def terminate(self):
        self.model.terminate()
