from typing import Dict
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification

import torch


class TaskVector:

    def __init__(self, pretrained_state_dict: Dict = None, finetuned_state_dict: Dict = None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_state_dict is not None and finetuned_state_dict is not None
            with torch.no_grad():

                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    @staticmethod
    def from_checkpoint(pretrained_checkpoint: str, finetuned_checkpoint: str) -> "TaskVector":
        """
        Load Task vector from two pytorch checkpoints
        """
        pretrained_state_dict = TaskVector.load_checkpoint(pretrained_checkpoint)
        finetuned_state_dict = TaskVector.load_checkpoint(finetuned_checkpoint)
        return TaskVector(pretrained_state_dict=pretrained_state_dict, finetuned_state_dict=finetuned_state_dict)

    @staticmethod
    def from_huggingface(pretrained_model_name: str, finetuned_model_name: str,
                         model_type: AutoModel = BertForSequenceClassification) -> "TaskVector":
        """
        Load Task vector from two saved hugging face model.
        """
        pretrained_state_dict = TaskVector.load_hugging_face(pretrained_model_name, model_type)
        finetuned_state_dict = TaskVector.load_hugging_face(finetuned_model_name, model_type)
        return TaskVector(pretrained_state_dict=pretrained_state_dict, finetuned_state_dict=finetuned_state_dict)

    @staticmethod
    def from_hugging_face_or_checkpoint(pretrained_path_or_name: str, finetuned_path_or_name: str,
                                        model_type: AutoModel = BertForSequenceClassification) -> "TaskVector":
        """
          Load Task vector from two saved hugging face model or a pytorch checkpoint.
        """
        pretrained_state_dict = TaskVector.load_hugging_face_or_checkpoint(pretrained_path_or_name, model_type)
        finetuned_state_dict = TaskVector.load_hugging_face_or_checkpoint(finetuned_path_or_name, model_type)
        return TaskVector(pretrained_state_dict=pretrained_state_dict, finetuned_state_dict=finetuned_state_dict)

    @staticmethod
    def load_hugging_face_or_checkpoint(path_or_name: str,
                                        model_type: AutoModel = BertForSequenceClassification) -> Dict:
        """
        Load either saved hugging face model or from checkpoint.
        """
        if path_or_name.endswith(".pt"):
            return TaskVector.load_checkpoint(path_or_name)
        else:
            return TaskVector.load_hugging_face(path_or_name, model_type)

    @staticmethod
    def load_hugging_face(model_name: str, model_type: AutoModel = BertForSequenceClassification) -> Dict:
        """
        Load saved hugging face model.
        """
        state_dict = model_type.from_pretrained(model_name).state_dict()
        return state_dict

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict:
        """
        Load pytorch checkpoints
        """
        state_dict = torch.load(checkpoint_path).state_dict()
        return state_dict

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint: str = None, pretrained_model_name: str = None, scaling_coef=1.0,
                 model_type: AutoModel = BertForSequenceClassification):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            if pretrained_checkpoint:
                pretrained_model = torch.load(pretrained_checkpoint)
            elif pretrained_model_name:
                pretrained_model = model_type.from_pretrained(pretrained_model_name)
            else:
                raise Exception("Must provide a pretrained checkpoint path or the model name to load from huggingface")
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
