import torch
from diffusers import AutoencoderKL, FluxTransformer2DModel
import time

threshold_mem = 9 * 1024 * 1024 * 1024 #9G
class ForwardHookManager:
    def __init__(self):
        self._registered_models = []
        self._origin_states = {}
        self._load_order = {}

    def _get_available_memory(self):
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        return total - reserved

    def _free_up_memory(self, required_mem, cache_model = None):
        # print("len(self._load_order):", len(self._load_order),  [model.__class__.__name__ for model in self._load_order.keys()])
        # print("Available memory:", self._get_available_memory())
        # print("required_mem:", required_mem)
        sorted_items = sorted(self._load_order.items(), key=lambda x: x[1])
        for model, value in sorted_items:
            if self._origin_states[model]['in_cuda']:
                if model != cache_model:
                    model.to(self._origin_states[model]['origin_device'])
                    self._origin_states[model]['in_cuda'] = False
                    del self._load_order[model]
                torch.cuda.empty_cache()
                if cache_model == None:
                    if self._get_available_memory() - threshold_mem >= required_mem:
                        return True
                else:
                    if self._get_available_memory() >= threshold_mem:
                        return True
        return False
    
    def model_to_cuda(self, model):
        if torch.cuda.is_available():
            origin_device = model.device if hasattr(model, 'device') else torch.device('cpu')
            if model not in self._registered_models:
                if torch.cuda.is_available():
                    if origin_device.type == 'cuda':
                        model.to('cpu')
                    with torch.no_grad():
                        ori_cuda_memory = torch.cuda.memory_allocated()
                        model.to("cuda")
                        cuda_memory = torch.cuda.memory_allocated() - ori_cuda_memory
                        model.to(origin_device)
                        torch.cuda.empty_cache()
                    
                    self._origin_states[model] = {
                        'origin_forward': model.forward,
                        'origin_device': origin_device,
                        'cuda_memory': cuda_memory,
                        'in_cuda': False,
                    }
            available_mem = self._get_available_memory()
            required_mem = self._origin_states[model]['cuda_memory']
            # print(f"required_mem: {required_mem}, available_mem: {available_mem}")
            if origin_device.type != 'cuda':
                if available_mem - threshold_mem < required_mem:
                    self._free_up_memory(required_mem)
                        # raise RuntimeError(f"Insufficient GPU memory. Required: {required_mem}, Available: {self._get_available_memory()}")
                model.to('cuda')
                if model not in self._load_order:
                    self._load_order[model] = 3
            else:
                if available_mem < threshold_mem:
                    self._free_up_memory(required_mem, model)
                        # raise RuntimeError(f"Insufficient GPU memory. Required: {required_mem}, Available: {self._get_available_memory()}")
            self._load_order[model] += 1
            for other_model in self._load_order:
                if other_model != model:
                    self._load_order[other_model] = max(self._load_order[other_model], 0)
            self._origin_states[model]['in_cuda'] = True
        return model

    def _register(self, model):
        if model not in self._registered_models:
            origin_device = model.device if hasattr(model, 'device') else torch.device('cpu')
            if torch.cuda.is_available():
                if origin_device.type == 'cuda':
                    model.to('cpu')
                with torch.no_grad():
                    ori_cuda_memory = torch.cuda.memory_allocated()
                    model.to("cuda")
                    cuda_memory = torch.cuda.memory_allocated() - ori_cuda_memory
                    model.to(origin_device)
                    torch.cuda.empty_cache()
                
                self._origin_states[model] = {
                    'origin_forward': model.forward,
                    'origin_device': origin_device,
                    'cuda_memory': cuda_memory,
                    'in_cuda': False,
                }

                def custom_forward(*args, **kwargs):      
                    if torch.cuda.is_available():
                        available_mem = self._get_available_memory()
                        required_mem = self._origin_states[model]['cuda_memory']
                        origin_device = model.device if hasattr(model, 'device') else self._origin_states[model]['origin_device']
                        if origin_device.type != 'cuda':
                            if available_mem - threshold_mem < required_mem:
                                self._free_up_memory(required_mem)
                            model.to('cuda')
                            
                            if model not in self._load_order:
                                self._load_order[model] = 3
                        else:
                            if self._get_available_memory() < threshold_mem:
                                self._free_up_memory(required_mem, model)
                        self._origin_states[model]['in_cuda'] = True
                        self._load_order[model] += 1
                        for other_model in self._load_order:
                            if other_model != model:
                                self._load_order[other_model] = max(self._load_order[other_model], 0)
                        args = tuple(arg.cuda() if isinstance(arg, torch.Tensor) and arg.device != 'cuda' else arg     
                                   for arg in args)
                        kwargs = {k: v.cuda() if isinstance(v, torch.Tensor) and v.device != 'cuda' else v 
                                for k, v in kwargs.items()}

                    result = self._origin_states[model]['origin_forward'](*args, **kwargs)
                    return result

                model.forward = custom_forward
                self._registered_models.append(model)
        return model

    def register(self, model):
        if isinstance(model, torch.nn.Module):
            if isinstance(model, AutoencoderKL):
                model.encoder = self._register(model.encoder)
                model.decoder = self._register(model.decoder)
            model = self._register(model)
        return model

    def revert(self):
        for model in self._registered_models:
            if model in self._origin_states:
                # model.forward = self._origin_states[model]['origin_forward']
                model.to(self._origin_states[model]['origin_device'])
        # self._registered_models.clear()
        # self._origin_states.clear()
        self._load_order.clear()
    

