from abc import ABC, abstractmethod
from ..Projectors import DTProjector
from ..Models import DTHfEncoder, LayerWrappebleDTHfLLM
from tqdm import tqdm
import torch

class DTTrainStratergy(ABC):
    
    llm : LayerWrappebleDTHfLLM 
    encoder : DTHfEncoder = None
    projector : DTProjector = None
    train_loader = None
    val_loader = None
    test_loader = None
    device = None

    def __init__(self, llm : LayerWrappebleDTHfLLM, encoder : DTHfEncoder, projector : DTProjector,
                 train_loader, val_loader,lr,device, 
                 target_layers : list[int]|int, enable_dp : bool = False, gpu_ids=None,**kwargs):
        
        self.llm = llm
        self.encoder = encoder
        self.projector = projector
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr

        self.device = device
        
        self.to()
        if enable_dp:
            self.enable_dp(gpu_ids)

        if isinstance(target_layers, int):
            target_layers = [target_layers]
        self.target_layers = target_layers
    
    def print_status(self):
        is_encoder_frozen = not any(p.requires_grad for p in self.encoder.model.parameters())
        is_projector_frozen = not any(p.requires_grad for p in self.projector.parameters())
        print(f"Encoder_Frozen: {is_encoder_frozen}")
        print(f"Projector_Frozen: {is_projector_frozen}")
        self.llm.print_status()
    
    def to(self, device = None):
        if device is None:
            device = self.device
        else:
            self.device = device
        
        if device is not None:
            self.llm.to(device)
            self.encoder.to(device)
            self.projector.to(device)
    
    def enable_dp(self, gpu_ids=None):
       
        if self.device == torch.device('cpu'):
            raise ValueError("Cannot enable DataParallel on CPU")
        
        self.to()
        self.encoder.enableDP(gpu_ids)
        self.llm.enableDP(gpu_ids)
        self.projector = torch.nn.DataParallel(self.projector, device_ids=gpu_ids)

    
    def freeze_encoder(self, freeze : bool = True):
        self.encoder.freeze(freeze)
    
    def read2train(self, layer_idx : int, engage_all : bool = False, engage_specific : list[int] = []):
        self.llm.ready2train(layer_idx, engage_all, engage_specific)
    
    def freeze_llm(self):
        self.llm.freeze()
    
    def freeze_projector(self, status : bool = True):
        for param in self.projector.parameters():
            param.requires_grad = not status

    @abstractmethod
    def train(self, epochs : int, **kwargs):
        pass

class TwoPhasedTS(DTTrainStratergy):

    def __init__(self, llm, encoder, projector, train_loader, val_loader, lr, device, target_layers,
                 mapper_train_loader, mapper_val_loader, layer_shift : int = 0 , enable_dp : bool = False, gpu_ids=None,
                 **kwargs):
        super().__init__(llm, encoder, projector, train_loader, val_loader, lr, device, target_layers, 
                            enable_dp, gpu_ids, **kwargs)
        
        self.mapper_train_loader = mapper_train_loader
        self.mapper_val_loader = mapper_val_loader

        self.projector_optimizer = torch.optim.Adam(self.projector.parameters(), lr=lr)
        self.criteria = torch.nn.MSELoss()
        self.layer_shift = layer_shift
    
    def compute_loss(self, projected_data, encoded_data):
        return self.criteria(projected_data, encoded_data)
    
    def train_projector(self, epochs : int, layer_idx ,**kwargs):
        
        self.llm.freeze()
        self.encoder.freeze()
        self.llm.engage_layer_wrapper(layer_idx, status=False)

        self.print_status()

        for epoch in range(epochs):
            
            self.projector.train()
            total_loss = 0.0

            for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                self.projector_optimizer.zero_grad()
                
                encoded_data = self.get_encoder_output(batch)
                llm_embeddings = self.llm_layer_output(batch, layer_idx)
                
                projected_data = self.projector(llm_embeddings)
                loss = self.compute_loss(projected_data, encoded_data)
                
                loss.backward()
                self.projector_optimizer.step()

                total_loss += loss.item()

            del llm_embeddings, projected_data, encoded_data 
            torch.cuda.empty_cache()
            
            total_loss /= len(self.train_loader)
            eval_loss = self.evaluate_projector(layer_idx, self.val_loader)
            print(f"Epoch {epoch} : Loss {total_loss} Eval Loss {eval_loss}")
    
    def evaluate_projector(self, layer_idx, loader):
        
        self.projector.eval()
        self.llm.freeze()
        self.encoder.freeze()


        total_loss = 0.0
        for i, batch in enumerate(tqdm(loader)):
            
            encoded_data = self.get_encoder_output(batch)
            llm_embeddings = self.llm_layer_output(batch, layer_idx)
            
            with torch.no_grad():
                projected_data = self.projector(llm_embeddings)
            loss = self.compute_loss(projected_data, encoded_data)
            total_loss += loss.item()
        
        del llm_embeddings, projected_data, encoded_data
        torch.cuda.empty_cache()

        total_loss /= len(loader)
        return total_loss

    def train_mapper(self, epochs : int, layer_idx, **kwargs):
        
        self.llm.freeze()
        self.encoder.freeze()
        self.freeze_projector()
        
        wrapper = self.llm.ready2train(layer_idx)
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=self.lr)
        
        self.print_status()

        
        for epoch in range(epochs):
            
            wrapper.train()
            self.projector.train()
            total_loss = 0.0

            for i, batch in enumerate(tqdm(self.mapper_train_loader, desc=f"Epoch {epoch}")):
                
                optimizer.zero_grad()
                
                
                encoded_data = self.get_encoder_output(batch)
                
                llm_embeddings = self.llm_layer_output(batch, layer_idx)

                projected_data = self.projector(llm_embeddings)
                
                loss = self.compute_loss(projected_data, encoded_data)
                loss.backward()
                
                optimizer.step()

                total_loss += loss.item()

            del llm_embeddings, projected_data, encoded_data
            torch.cuda.empty_cache()

            total_loss /= len(self.mapper_train_loader)
            eval_loss = self.evaluate_mapper(layer_idx, self.mapper_val_loader)
            print(f"Epoch {epoch} : Loss {total_loss} Eval Loss {eval_loss}")
    
    def evaluate_mapper(self, layer_idx, loader):
        
        self.llm.freeze()
        self.encoder.freeze()
        self.freeze_projector()
        
        wrapper = self.llm.ready2train(layer_idx)

        wrapper.eval()
        self.projector.eval()
        
        total_loss = 0.0
        for i, batch in enumerate(tqdm(loader)):
            
            encoded_data = self.get_encoder_output(batch)
            llm_embeddings = self.llm_layer_output(batch, layer_idx)
            
            with torch.no_grad():
                projected_data = self.projector(llm_embeddings)
            loss = self.compute_loss(projected_data, encoded_data)
            total_loss += loss.item()
        
        del llm_embeddings, projected_data, encoded_data
        torch.cuda.empty_cache()
        total_loss /= len(loader)
        return total_loss
    
    def train(self, epochs : int, **kwargs):
        
        for layer_idx in self.target_layers:
            self.train_projector(epochs, layer_idx)
            self.train_mapper(epochs, layer_idx)
    
    def get_encoder_output(self, text):
        return self.encoder.encode(text, "mean")
    
    def llm_layer_output(self, text, layer_idx):
        layer_idx += self.layer_shift
        return self.llm.get_Layer_output(text, layer_idx, "mean")

class TwoPhasedSeq2SeqTS(TwoPhasedTS):

    def __init__(self, llm, encoder, projector, train_loader, val_loader, lr, device, target_layers, mapper_train_loader, mapper_val_loader, layer_shift = 0, enable_dp = False, gpu_ids=None, **kwargs):
        super().__init__(llm, encoder, projector, train_loader, val_loader, lr, device, target_layers, mapper_train_loader, mapper_val_loader, layer_shift, enable_dp, gpu_ids, **kwargs)
    
    def get_encoder_output(self, text):
        return self.encoder.encode(text, "mean")
    
    def llm_layer_output(self, text, layer_idx):
        layer_idx += self.layer_shift
        return self.llm.get_Layer_output(text, layer_idx)

class TwoPhasedSeq2SeqKL(TwoPhasedSeq2SeqTS):

    def __init__(self, llm, encoder, projector, train_loader, val_loader, lr, device, target_layers, mapper_train_loader, mapper_val_loader, layer_shift = 0, enable_dp = False, gpu_ids=None, **kwargs):
        super().__init__(llm, encoder, projector, train_loader, val_loader, lr, device, target_layers, mapper_train_loader, mapper_val_loader, layer_shift, enable_dp, gpu_ids, **kwargs)

        self.criteria = torch.nn.KLDivLoss(reduction="batchmean")
    
    def compute_loss(self, projected_data, encoded_data):

        probs_projected = torch.nn.functional.log_softmax(projected_data, dim=1)
        probs_encoded = torch.nn.functional.softmax(encoded_data, dim=1)

        return self.criteria(probs_projected, probs_encoded)

class MixedTS(DTTrainStratergy):

    def __init__(self, llm, encoder, projector, train_loader, val_loader, lr, device, target_layers,**kwargs):
        super().__init__(llm, encoder, projector, train_loader, val_loader, lr, device, target_layers, **kwargs)

        self.criteria = torch.nn.MSELoss()
        
    def train_layer(self, epochs, layer_idx):
        
        self.llm.freeze()
        self.encoder.freeze()
        wrapper = self.llm.ready2train(layer_idx)

        optimizer = torch.optim.Adam(list(wrapper.parameters()) + list(self.projector.parameters()), lr=self.lr)
        self.llm.print_status()

        for epoch in range(epochs):
            
            self.projector.train()
            wrapper.train()

            total_loss = 0.0

            for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                optimizer.zero_grad()
                
                
                encoded_data = self.encoder.encode(batch,"mean")
                llm_embeddings = self.llm.get_Layer_output(batch, layer_idx, "mean")
    
                projected_data = self.projector(llm_embeddings)
                
                loss = self.criteria(projected_data, encoded_data)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            del llm_embeddings, projected_data, encoded_data
            torch.cuda.empty_cache()
            
            total_loss /= len(self.train_loader)
            eval_loss = self.eval_layer(layer_idx, self.val_loader)
            print(f"Epoch {epoch} : Loss {total_loss} Eval Loss {eval_loss}")
    
    def eval_layer(self, layer_idx, loader):
        
        self.llm.freeze()
        self.encoder.freeze()
        wrapper = self.llm.ready2train(layer_idx)

        wrapper.eval()

        total_loss = 0.0

        for i, batch in enumerate(tqdm(loader)):
            
            encoded_data = self.encoder.encode(batch,"mean")
            llm_embeddings = self.llm.get_Layer_output(batch, layer_idx, "mean")
            
            with torch.no_grad():
                mapped_data = wrapper(llm_embeddings)
                projected_data = self.projector(mapped_data)
            loss = self.criteria(projected_data, encoded_data)
            total_loss += loss.item()
        
        total_loss /= len(loader)
        return total_loss
    
    def train(self, epochs : int, **kwargs):
        
        for layer_idx in self.target_layers:
            self.train_layer(epochs, layer_idx)