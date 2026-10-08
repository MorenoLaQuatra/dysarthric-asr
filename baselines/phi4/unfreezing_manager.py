"""
Model parameter unfreezing manager for systematic dysarthric ASR experiments.
"""

import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class UnfreezingConfig:
    """Configuration for model parameter unfreezing strategies"""
    
    # Core components
    unfreeze_lora_speech: bool = True
    unfreeze_audio_components: bool = False
    unfreeze_token_embeddings: bool = False
    
    # Language model components
    unfreeze_language_layers: bool = False
    language_layer_strategy: str = "none"  # "first_n", "last_n", "middle_n", "specific", "all"
    language_layer_count: int = 4
    language_layer_indices: List[int] = None
    
    # Additional components
    unfreeze_layer_norms: bool = False
    unfreeze_model_norm: bool = False
    unfreeze_lm_head: bool = False
    
    # Safety settings
    max_trainable_percentage: float = 50.0
    
    def __post_init__(self):
        if self.language_layer_indices is None:
            self.language_layer_indices = []


class UnfreezingManager:
    """Manages systematic parameter unfreezing for multimodal models"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from config dictionary with dot notation support"""
        self.config = self._build_config(config_dict)
        
    def _build_config(self, config_dict: Dict[str, Any]) -> UnfreezingConfig:
        """Build UnfreezingConfig from nested config dictionary"""
        
        # Core components
        unfreeze_lora_speech = config_dict.get('unfreeze_lora_speech', True)
        unfreeze_audio_components = config_dict.get('unfreeze_audio_components', False)
        unfreeze_token_embeddings = config_dict.get('unfreeze_token_embeddings', False)
        
        # Language model components
        unfreeze_language_layers = config_dict.get('unfreeze_language_layers', False)
        language_layer_strategy = config_dict.get('language_layer_strategy', 'none')
        language_layer_count = config_dict.get('language_layer_count', 4)
        language_layer_indices = config_dict.get('language_layer_indices', [])
        
        # Additional components
        unfreeze_layer_norms = config_dict.get('unfreeze_layer_norms', False)
        unfreeze_model_norm = config_dict.get('unfreeze_model_norm', False)
        unfreeze_lm_head = config_dict.get('unfreeze_lm_head', False)
        
        # Safety settings
        max_trainable_percentage = config_dict.get('max_trainable_percentage', 50.0)
        
        return UnfreezingConfig(
            unfreeze_lora_speech=unfreeze_lora_speech,
            unfreeze_audio_components=unfreeze_audio_components,
            unfreeze_token_embeddings=unfreeze_token_embeddings,
            unfreeze_language_layers=unfreeze_language_layers,
            language_layer_strategy=language_layer_strategy,
            language_layer_count=language_layer_count,
            language_layer_indices=language_layer_indices,
            unfreeze_layer_norms=unfreeze_layer_norms,
            unfreeze_model_norm=unfreeze_model_norm,
            unfreeze_lm_head=unfreeze_lm_head,
            max_trainable_percentage=max_trainable_percentage,
        )
    
    def apply_unfreezing_strategy(self, model, total_layers: int = 32) -> Dict[str, Any]:
        """Apply unfreezing strategy to model based on configuration"""
        
        # First, freeze everything
        for param in model.parameters():
            param.requires_grad = False
        
        stats = {
            "components_unfrozen": [],
            "layers_unfrozen": [],
            "safety_triggered": False,
            "total_trainable": 0,
            "total_params": 0,
            "trainable_percentage": 0.0
        }
        
        # 1. LoRA Speech adapters
        if self.config.unfreeze_lora_speech:
            unfrozen_count = self._unfreeze_lora_speech(model)
            if unfrozen_count > 0:
                stats["components_unfrozen"].append(f"lora_speech ({unfrozen_count} params)")
        
        # 2. Audio components
        if self.config.unfreeze_audio_components:
            unfrozen_count = self._unfreeze_audio_components(model)
            if unfrozen_count > 0:
                stats["components_unfrozen"].append(f"audio_components ({unfrozen_count} params)")
        
        # 3. Token embeddings
        if self.config.unfreeze_token_embeddings:
            success = self._unfreeze_token_embeddings(model)
            if success:
                stats["components_unfrozen"].append("token_embeddings")
        
        # 4. Language model layers
        if self.config.unfreeze_language_layers:
            layer_stats = self._unfreeze_language_layers(model, total_layers)
            stats["layers_unfrozen"].extend(layer_stats)
        
        # 5. Model norm
        if self.config.unfreeze_model_norm:
            success = self._unfreeze_model_norm(model)
            if success:
                stats["components_unfrozen"].append("model_norm")
        
        # 6. LM head
        if self.config.unfreeze_lm_head:
            success = self._unfreeze_lm_head(model)
            if success:
                stats["components_unfrozen"].append("lm_head")
        
        # Calculate final statistics and safety check
        stats.update(self._calculate_stats(model))
        
        # Safety check
        if stats["trainable_percentage"] > self.config.max_trainable_percentage:
            print(f"WARNING: {stats['trainable_percentage']:.1f}% parameters trainable "
                  f"(limit: {self.config.max_trainable_percentage}%)")
            stats["safety_triggered"] = True
        
        return stats
    
    def _unfreeze_lora_speech(self, model) -> int:
        """Unfreeze LoRA speech adapters"""
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name and 'speech' in name:
                param.requires_grad = True
                unfrozen_count += 1
        return unfrozen_count
    
    def _unfreeze_audio_components(self, model) -> int:
        """Unfreeze audio components (encoder, projection, embed)"""
        unfrozen_count = 0
        try:
            audio_embed = model.model.embed_tokens_extend.audio_embed
            for param in audio_embed.parameters():
                param.requires_grad = True
                unfrozen_count += 1
        except AttributeError:
            print("Could not find audio components")
        return unfrozen_count
    
    def _unfreeze_token_embeddings(self, model) -> bool:
        """Unfreeze token embeddings"""
        try:
            model.model.embed_tokens.weight.requires_grad = True
            return True
        except AttributeError:
            print("Could not find token embeddings")
            return False
    
    def _unfreeze_language_layers(self, model, total_layers: int) -> List[str]:
        """Unfreeze language model layers based on strategy"""
        layer_indices = self._get_layer_indices(total_layers)
        layer_stats = []
        
        for layer_idx in layer_indices:
            try:
                layer = model.model.layers[layer_idx]
                unfrozen_params = 0
                
                # Unfreeze base layer components (not LoRA adapters)
                for name, param in layer.named_parameters():
                    if 'base_layer' in name:
                        param.requires_grad = True
                        unfrozen_params += 1
                    # Optionally unfreeze layer norms
                    elif (self.config.unfreeze_layer_norms and 
                          ('layernorm' in name.lower() or 'layer_norm' in name.lower())):
                        param.requires_grad = True
                        unfrozen_params += 1
                
                if unfrozen_params > 0:
                    layer_stats.append(f"layer_{layer_idx} ({unfrozen_params} params)")
                    
            except (AttributeError, IndexError):
                print(f"Could not unfreeze layer {layer_idx}")
        
        return layer_stats
    
    def _unfreeze_model_norm(self, model) -> bool:
        """Unfreeze model norm"""
        try:
            if hasattr(model.model, 'norm'):
                model.model.norm.weight.requires_grad = True
                return True
        except AttributeError:
            print("Could not find model norm")
        return False
    
    def _unfreeze_lm_head(self, model) -> bool:
        """Unfreeze LM head"""
        try:
            if hasattr(model, 'lm_head'):
                model.lm_head.weight.requires_grad = True
                return True
            elif hasattr(model.model, 'lm_head'):
                model.model.lm_head.weight.requires_grad = True
                return True
        except AttributeError:
            print("Could not find LM head")
        return False
    
    def _get_layer_indices(self, total_layers: int) -> List[int]:
        """Get layer indices based on strategy"""
        strategy = self.config.language_layer_strategy
        
        if strategy == "none":
            return []
        elif strategy == "all":
            return list(range(total_layers))
        elif strategy == "first_n":
            return list(range(min(self.config.language_layer_count, total_layers)))
        elif strategy == "last_n":
            start_idx = max(0, total_layers - self.config.language_layer_count)
            return list(range(start_idx, total_layers))
        elif strategy == "middle_n":
            start_idx = max(0, (total_layers - self.config.language_layer_count) // 2)
            end_idx = min(total_layers, start_idx + self.config.language_layer_count)
            return list(range(start_idx, end_idx))
        elif strategy == "specific":
            return [idx for idx in self.config.language_layer_indices if 0 <= idx < total_layers]
        else:
            raise ValueError(f"Unknown language layer strategy: {strategy}")
    
    def _calculate_stats(self, model) -> Dict[str, Any]:
        """Calculate parameter statistics"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = (trainable_params / total_params) * 100
        
        return {
            "total_trainable": trainable_params,
            "total_params": total_params,
            "trainable_percentage": trainable_percentage
        }
    
    def print_unfreezing_stats(self, stats: Dict[str, Any]):
        """Print detailed statistics about unfreezing configuration"""
        print(f"\n{'='*80}")
        print(f"UNFREEZING STRATEGY APPLIED")
        print(f"{'='*80}")
        
        # Configuration summary
        print(f"CONFIGURATION:")
        print(f"   LoRA Speech: {'✓' if self.config.unfreeze_lora_speech else '✗'}")
        print(f"   Audio Components: {'✓' if self.config.unfreeze_audio_components else '✗'}")
        print(f"   Token Embeddings: {'✓' if self.config.unfreeze_token_embeddings else '✗'}")
        print(f"   Language Layers: {'✓' if self.config.unfreeze_language_layers else '✗'}")
        if self.config.unfreeze_language_layers:
            print(f"      Strategy: {self.config.language_layer_strategy}")
            if self.config.language_layer_strategy in ["first_n", "last_n", "middle_n"]:
                print(f"      Count: {self.config.language_layer_count}")
            elif self.config.language_layer_strategy == "specific":
                print(f"      Indices: {self.config.language_layer_indices}")
        print(f"   Model Norm: {'✓' if self.config.unfreeze_model_norm else '✗'}")
        print(f"   LM Head: {'✓' if self.config.unfreeze_lm_head else '✗'}")
        
        # Statistics
        print(f"\nSTATISTICS:")
        print(f"   Total parameters: {stats['total_params']:,} ({stats['total_params']/1e6:.1f}M)")
        print(f"   Trainable: {stats['total_trainable']:,} ({stats['total_trainable']/1e6:.1f}M)")
        print(f"   Percentage: {stats['trainable_percentage']:.2f}%")
        
        if stats['safety_triggered']:
            print(f"   WARNING: Safety limit exceeded!")
        
        # Components unfrozen
        print(f"\nUNFROZEN COMPONENTS:")
        for component in stats['components_unfrozen']:
            print(f"   ✓ {component}")
        
        if stats['layers_unfrozen']:
            print(f"\nUNFROZEN LAYERS:")
            for layer in stats['layers_unfrozen']:
                print(f"   ✓ {layer}")
        
        print(f"{'='*80}")