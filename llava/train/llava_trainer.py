import os
import torch
import torch.nn as nn
import json
from transformers import Trainer
from typing import Dict, Optional, Sequence


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class LLaVATrainer(Trainer):

    ## ################## May 29th, new function for saving the teacher logits ####################

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        This version also saves the logits for each input during training.
        """
        
        ################ ####### Code changes from original transformers compute loss ######### ################ 
        logit_save_path = "/scratch/ae2195/PPTP-KD-LLaVA-Med/llava/train/logits.json"  # Define your logit save path here

        # Initialize the JSON file if it does not exist
        if logit_save_path and not os.path.exists(logit_save_path):
            with open(logit_save_path, 'w') as f:
                json.dump([], f)
        
        ################ #### End of Changes ############ ################ 

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        ################ ###### Code changes from original transformers compute loss  ########## ################ 
       
        # Extract and save logits for each input
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]  # Depending on the model output format
        if logit_save_path:
            input_ids = inputs["input_ids"]
            logits_to_save = []
            for i, logit in enumerate(logits):
                logit_dict = {
                    "input_id": input_ids[i].cpu().tolist(),
                    "logits": logit.cpu().tolist()
                }
                logits_to_save.append(logit_dict)
            
            # Append new logits to the JSON file
            try:
                with open(logit_save_path, 'r+') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        # If the file is empty or invalid, initialize it
                        existing_data = []
                    existing_data.extend(logits_to_save)
                    f.seek(0)
                    json.dump(existing_data, f)
            except FileNotFoundError:
                # If the file does not exist, create and initialize it
                with open(logit_save_path, 'w') as f:
                    json.dump(logits_to_save, f)

        ################ ######  End of Changes ########## ################ 

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v.cpu().clone().detach() # Chunyuan: to solve the saving OOM problem 

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        super(LLaVATrainer, self)._save(output_dir, state_dict)
