import os
import torch
import torch.nn as nn
import json
import h5py
from transformers import Trainer
from typing import Dict, Optional, Sequence
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (torch.nn.Module): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class LLaVATrainer(Trainer):
    ####METHOD 3: USIGN DATASET PACKAGE
    def __init__(self, model, tokenizer, *args, **kwargs):
        # Initialize the Trainer with only the arguments it expects
        super().__init__(model=model, tokenizer=tokenizer, *args, **kwargs)
        # Then handle the teacher_model separately for the LLaVATrainer
        logit_save_path = "/scratch/ltl2113/PPTP-KD-LLaVA-Med/llava/train/Dataset_logits" 
        self.logit_save_path = logit_save_path
        # Initialize an empty dataset if it doesn't exist
        if not os.path.exists(logit_save_path):
            self.logits_dataset = Dataset.from_dict({"input_id": [], "logits": []})
        else:
            pass

    def compute_loss(self, model, inputs, return_outputs=False):
        """pip
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        This version also saves the logits for each input during training.
        """
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Extract and save logits for each input
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]  # Depending on the model output format
        

        input_ids = inputs["input_ids"]

        # Ensure input_ids and logits are on the same device
        device = input_ids.device
        logits = logits.to(device)

        # Get the top k logits and their indices
        top_k = 100  # Change this to the desired number of top logits
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
        # print(f"Logits size: {top_logits.shape}")

        # Convert tensors to lists for saving
        input_ids_list = input_ids.cpu().tolist()
        top_logits_list = top_logits.cpu().tolist()
        top_indices_list = top_indices.cpu().tolist()

        # Prepare new data to be added
        new_data = {
            "input_id": input_ids_list,
            "top_logits": top_logits_list,
            "top_indices": top_indices_list
        }
        new_dataset = Dataset.from_dict(new_data)
        
        # Concatenate the new dataset with the existing dataset
        self.logits_dataset = concatenate_datasets([self.logits_dataset, new_dataset])

        # Save the dataset to disk
        self.logits_dataset.save_to_disk(self.logit_save_path)

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
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    ## ####METHOD 1:  May 29th,saving the teacher logits IN JSON FORMAT ####################

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     This version also saves the logits for each input during training.
    #     """
        
    #     ################ ####### Code changes from original transformers compute loss ######### ################ 
    #     logit_save_path = "/scratch/ltl2113/PPTP-KD-LLaVA-Med/llava/train/logits.json"  # Define your logit save path here

    #     # Initialize the JSON file if it does not exist
    #     if logit_save_path and not os.path.exists(logit_save_path):
    #         with open(logit_save_path, 'w') as f:
    #             json.dump([], f)
        
    #     ################ #### End of Changes ############ ################ 

    #     if self.label_smoother is not None and "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
    #     outputs = model(**inputs)
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]

    #     ################ ###### Code changes from original transformers compute loss  ########## ################ 
    #     # Extract and save logits for each input
    #     logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]  # Depending on the model output format
    #     # Print the size of the logits
    #     print(f"Logits size: {logits.shape}")
    #     if logit_save_path:
    #         input_ids = inputs["input_ids"]
    #         with open(logit_save_path, 'a') as f:  # Open file in append mode
    #             for i, logit in enumerate(logits):
    #                 logit_dict = {
    #                     "input_id": input_ids[i].cpu().tolist(),
    #                     "logits": logit.cpu().tolist()
    #                 }
    #                 # print("Current logit: ",logit_dict)
    #                 f.write(json.dumps(logit_dict) + '\n')  # Write each logit set as a new line

    #     ################ ######  End of Changes ########## ################ 

    #     if labels is not None:
    #         unwrapped_model = self.accelerator.unwrap_model(model)
    #         if _is_peft_model(unwrapped_model):
    #             model_name = unwrapped_model.base_model.model._get_name()
    #         else:
    #             model_name = unwrapped_model._get_name()
    #         if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #             loss = self.label_smoother(outputs, labels, shift_labels=True)
    #         else:
    #             loss = self.label_smoother(outputs, labels)
    #     else:
    #         if isinstance(outputs, dict) and "loss" not in outputs:
    #             raise ValueError(
    #                 "The model did not return a loss from the inputs, only the following keys: "
    #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #             )
    #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     return (loss, outputs) if return_outputs else loss

    # ## #############METHOD 2: MAY 30th: SAVING IN H5PY FORMAT ##########
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     This version also saves the logits for each input during training.
    #     """
        
    #     ################ ####### Code changes from original transformers compute loss ######### ################ 
    #     logit_save_path = "/scratch/ltl2113/PPTP-KD-LLaVA-Med/llava/train/logits/logits_compress_6.h5"  # Define your logit save path here

    #     # Initialize the HDF5 file if it does not exist
    #     if logit_save_path and not os.path.exists(logit_save_path):
    #         with h5py.File(logit_save_path, 'w') as f:
    #             f.create_group("logits")

    #     ################ #### End of Changes ############ ################ 

    #     if self.label_smoother is not None and "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
    #     outputs = model(**inputs)
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]

    #     ################ ###### Code changes from original transformers compute loss  ########## ################ 
    #     # Extract and save logits for each input
    #     logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]  # Depending on the model output format
    #     # Print the size of the logits
    #     print(f"Logits size: {logits.shape}")
    #     if logit_save_path:
    #         input_ids = inputs["input_ids"]
    #         with h5py.File(logit_save_path, 'a') as f:  # Open file in append mode
    #             logit_group = f["logits"]
    #             for i, logit in enumerate(logits):
    #                 input_id_list = input_ids[i].cpu().tolist()
    #                 logit_list = logit.cpu().tolist()
    #                 # Use the input_id as the group name
    #                 group_name = '-'.join(map(str, input_id_list))  # Create a unique group name using the input IDs
    #                 if group_name in logit_group:
    #                     # Generate a unique name for the group if it already exists
    #                     suffix = 1
    #                     while f"{group_name}_{suffix}" in logit_group:
    #                         suffix += 1
    #                     group_name = f"{group_name}_{suffix}"
    #                 data_group = logit_group.create_group(group_name)
    #                 data_group.create_dataset("input_id", data=input_id_list, compression="gzip", compression_opts=6)
    #                 data_group.create_dataset("logits", data=logit_list, compression="gzip", compression_opts=6)
    #     ################ ######  End of Changes ########## ################ 

    #     if labels is not None:
    #         unwrapped_model = self.accelerator.unwrap_model(model)
    #         if _is_peft_model(unwrapped_model):
    #             model_name = unwrapped_model.base_model.model._get_name()
    #         else:
    #             model_name = unwrapped_model._get_name()
    #         if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #             loss = self.label_smoother(outputs, labels, shift_labels=True)
    #         else:
    #             loss = self.label_smoother(outputs, labels)
    #     else:
    #         if isinstance(outputs, dict) and "loss" not in outputs:
    #             raise ValueError(
    #                 "The model did not return a loss from the inputs, only the following keys: "
    #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #             )
    #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     return (loss, outputs) if return_outputs else loss

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
