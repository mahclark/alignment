import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Callable
from tabulate import tabulate
from bokeh.plotting import figure, curdoc
from bokeh.models import LinearColorMapper, ColorBar
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum, auto
from dataclasses import dataclass


@dataclass
class AggregationType:
    agg: Callable[[Tensor, int], Tensor]


class Aggregation(Enum):
    mean = AggregationType(torch.mean)
    max = AggregationType(lambda tensor, dim: torch.max(tensor, dim=dim).values)

    def __init__(self, agg_type: AggregationType):
        self._agg = agg_type.agg

    def __str__(self) -> str:
        return f'[{self.name}]'

    @property
    def agg(self):
        return self._agg


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
        """Returns the layer and head with the largest attention between the two tokens at the indices provided"""
        argmax = self.attention_weights[:, :, token_index, target_token_index].argmax()
        layer, head = torch.unravel_index(argmax, self.attention_weights.shape[:2])
        return layer.item(), head.item()
    
    def get_mean_attention(self, *, token_index: int, layer: Optional[int] = None, head: Optional[int] = None) -> torch.Tensor:
        if layer is None and head is None:
            return self.attention_weights[:, :, token_index, :].mean(dim=[0,1])
        if layer is None:
            return self.attention_weights[:, head, token_index, :].mean(dim=0)
        if head is None:
            return self.attention_weights[layer, :, token_index, :].mean(dim=0)
        
        return self.attention_weights[layer, head, token_index, :]
    
    def get_max_attention(self, *, token_index: int, layer: Optional[int] = None, head: Optional[int] = None) -> torch.Tensor:
        if layer is None and head is None:
            return torch.amax(self.attention_weights[:, :, token_index, :], dim=(0, 1))
        if layer is None:
            return torch.amax(self.attention_weights[:, head, token_index, :], dim=0)
        if head is None:
            return torch.amax(self.attention_weights[layer, :, token_index, :], dim=0)
        
        return self.attention_weights[layer, head, token_index, :]
    
    def print_attention(self, *, token_index: int, layer: Optional[int] = None, head: Optional[int] = None) -> None:
        attention = self.get_mean_attention(token_index=token_index, layer=layer, head=head)
        print(tabulate([
            self.tokens,
            [f'{n:.2f}' for n in attention.tolist()],
        ]))
    
    def heatmap_attention(self, *, token_index: int, layer: Optional[Union[int, Aggregation]] = None, head: Optional[Union[int, Aggregation]] = None) -> figure:
        if layer is None and head is None:
            raise ValueError('layer or head must be specified')
        
        if type(layer) == Aggregation and type(head) == Aggregation:
            raise ValueError('Using aggregations across 2 dimensions is undefined')
        
        if layer is None:
            if type(head) == Aggregation:
                attention = head.agg(self.attention_weights[:, :, token_index, :], dim=1)
            else:
                attention = self.attention_weights[:, head, token_index, :]
    
            title = f'Head {head}'
            y_label = 'Layer'

        elif head is None:
            if type(layer) == Aggregation:
                attention = layer.agg(self.attention_weights[:, :, token_index, :], dim=0)
            else:
                attention = self.attention_weights[layer, :, token_index, :]

            title = f'Layer {layer}'
            y_label = 'Head'

        else:
            if type(head) == Aggregation:
                attention = head.agg(self.attention_weights[layer, :, token_index, :], dim=1).unsqueeze(0)
            elif type(layer) == Aggregation:
                attention = layer.agg(self.attention_weights[:, head, token_index, :], dim=0).unsqueeze(0)
            else:
                attention = self.attention_weights[layer, head, token_index, :].unsqueeze(0)
            
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
    
    def token_intensities(
            self,
            *,
            token_index: int,
            layer: Optional[int] = None,
            head: Optional[int] = None,
            agg: Aggregation = Aggregation.mean,
        ) -> List[Tuple[str, float]]:

        if agg == Aggregation.mean:
            attention = self.get_mean_attention(token_index=token_index, layer=layer, head=head)
        elif agg == Aggregation.max:
            attention = self.get_max_attention(token_index=token_index, layer=layer, head=head)
        else:
            raise ValueError(f'Unknown aggregation {agg}')
        return list(zip(self.tokens, attention.tolist()))

    @property
    def space_token(self) -> str:
        token_ids = self.tokenizer.encode(' ')
        return self.tokenizer.convert_ids_to_tokens(token_ids)[-1]
    
    def token_intensity_html(
            self, 
            *,
            token_index: int,
            layer: Optional[int] = None,
            head: Optional[int] = None,
            agg: Aggregation = Aggregation.mean,
            style: Dict[str, str] = {},
            background_color: Tuple[int, int, int] = (26, 26, 26),
            highlight_color: Tuple[int, int, int] = (209, 69, 69),
        ) -> str:
        
        style = dict(
            {
                'color': 'rgb(213, 213, 213)',
                'background-color': f'rgb{background_color}',
                'font-size': '20px',
                'font-family': 'monospace',
            },
            **style
        )

        def highlight(text: str, intensity: float) -> str:
            bg_color = (
                int(background_color[0] + (highlight_color[0] - background_color[0]) * intensity),
                int(background_color[1] + (highlight_color[1] - background_color[1]) * intensity),
                int(background_color[2] + (highlight_color[2] - background_color[2]) * intensity),
            )
            return f"<span style='background-color: rgb{bg_color};'>{text}</span>"
        

        split_intensities = []
        for token, intensity in self.token_intensities(token_index=token_index, layer=layer, head=head, agg=agg):
            if token.startswith(self.space_token):
                split_intensities.append((' ', 0))
            split_intensities.append((token.lstrip(self.space_token), intensity))
                
        return (
            f"<span style='{' '.join([f'{k}: {v};' for k, v in style.items()])}'>"
            + ''.join([
                highlight(token.replace('<', '&lt;').replace('>', '&gt;'), intensity)
                for token, intensity in split_intensities
            ])
            + "</span>"
        )
