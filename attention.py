import numpy as np
from typing import Optional, Tuple, List
from tabulate import tabulate
from bokeh.plotting import figure, curdoc
from bokeh.models import LinearColorMapper, ColorBar
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class AttentionVisualizer:
    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            prompt: str,
            device: Optional[str] = None,
        ):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        if model.device != device:
            model = model.to(device)

        self.model = model
        self.tokenizer = tokenizer

        self._inputs = tokenizer(prompt, return_tensors='pt').to(device)
        self.tokens: List[str] = [
            token
            for token in tokenizer.convert_ids_to_tokens(self._inputs['input_ids'].squeeze(0).tolist())
        ]

        outputs = model(**self._inputs, output_attentions=True)
        self._output_attention: Tuple[torch.Tensor, ...] = outputs.attentions
        self._generated_text: Optional[str] = None

    @property
    def generated_text(self) -> str:
        if self._generated_text is not None:
            return self._generated_text

        output_sequences = self.model.generate(
            input_ids=self._inputs['input_ids'],
            max_length=50,
            temperature=1,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )

        self._generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return self._generated_text
    
    @property
    def attention_weights(self) -> torch.Tensor:
        return torch.stack(self._output_attention).squeeze(1).detach().cpu()
    
    def argmax_attention(self, *, token_index: int, target_token_index: int) -> Tuple[int, int]:
        argmax = self.attention_weights[:, :, token_index, target_token_index].argmax()
        layer, head = torch.unravel_index(argmax, self.attention_weights.shape[:2])
        print(f'''{
            self.tokens[token_index]
        }-{
            self.tokens[target_token_index]
        } max at layer {layer} head {head}, value {
            self.attention_weights[layer, head, token_index, target_token_index]
        }''')
        return layer.item(), head.item()
    
    def get_mean_attention(self, *, token_index: int, layer: Optional[int] = None, head: Optional[int] = None) -> torch.Tensor:
        if layer is None and head is None:
            return self.attention_weights[:, :, token_index, :].mean(dim=[0,1])
        if layer is None:
            return self.attention_weights[:, head, token_index, :].mean(dim=0)
        if head is None:
            return self.attention_weights[layer, :, token_index, :].mean(dim=0)
        
        return self.attention_weights[layer, head, token_index, :]
    
    def print_attention(self, *, token_index: int, layer: Optional[int] = None, head: Optional[int] = None) -> None:
        attention = self.get_mean_attention(token_index=token_index, layer=layer, head=head)
        print(tabulate([
            self.tokens,
            [f'{n:.2f}' for n in attention.tolist()],
        ]))
    
    def get_attention_for_head(self, *, token_index: int, head: int) -> torch.Tensor:
        return self.attention_weights[:, head, token_index, :]
    
    def get_attention_for_layer(self, *, token_index: int, layer: int) -> torch.Tensor:
        return self.attention_weights[layer, :, token_index, :]
    
    def get_attention_for_layer_and_head(self, *, token_index: int, layer: int, head: int) -> torch.Tensor:
        return self.attention_weights[layer, head, token_index, :]
    
    def heatmap_attention(self, *, token_index: int, layer: Optional[int] = None, head: Optional[int] = None) -> figure:
        if layer is None and head is None:
            raise ValueError('layer or head must be specified')
        
        if layer is None:
            attention = self.get_attention_for_head(token_index=token_index, head=head)
            title = f'Head {head}'
            y_label = 'Layer'
        elif head is None:
            attention = self.get_attention_for_layer(token_index=token_index, layer=layer)
            title = f'Layer {layer}'
            y_label = 'Head'
        else:
            attention = self.get_attention_for_layer_and_head(token_index=token_index, layer=layer, head=head).unsqueeze(0)
            title = f'Layer {layer} Head {head}'
            y_label = ''
        
        attention = attention.numpy()
        
        curdoc().theme = 'carbon'
        p = figure(
            width=500,
            height=400,
            x_range=self.tokens,#(0, attention.shape[1]),
            y_range=(0, attention.shape[0]),
            title=title,
            y_axis_label=y_label
        )

        color_mapper = LinearColorMapper(palette='Viridis256', low=attention.min(), high=attention.max())
        p.image(image=[attention], x=0, y=0, dw=attention.shape[1], dh=attention.shape[0], color_mapper=color_mapper)

        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
        p.add_layout(color_bar, 'right')

        p.xaxis.major_label_orientation = np.pi / 4

        return p
    
    def token_intensities(self, *, token_index: int, layer: Optional[int] = None, head: Optional[int] = None) -> List[Tuple[str, float]]:
        attention = self.get_mean_attention(token_index=token_index, layer=layer, head=head).tolist()
        return list(zip(self.tokens, attention))
