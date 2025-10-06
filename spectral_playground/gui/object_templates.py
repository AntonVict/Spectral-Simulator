"""Object template system for multi-fluorophore compositions."""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import List, Dict, Any
import json
from pathlib import Path


@dataclass
class FluorophoreComponent:
    """A single fluorophore component in a composition."""
    fluor_index: int
    ratio: float
    ratio_noise: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fluor_index': self.fluor_index,
            'ratio': self.ratio,
            'ratio_noise': self.ratio_noise
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> FluorophoreComponent:
        return FluorophoreComponent(
            fluor_index=data['fluor_index'],
            ratio=data['ratio'],
            ratio_noise=data.get('ratio_noise', 0.0)
        )


@dataclass
class ObjectTemplate:
    """Template for creating objects with specific fluorophore compositions."""
    name: str
    description: str = ""
    composition: List[FluorophoreComponent] = dataclass_field(default_factory=list)
    normalize: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'composition': [c.to_dict() for c in self.composition],
            'normalize': self.normalize
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ObjectTemplate:
        return ObjectTemplate(
            name=data['name'],
            description=data.get('description', ''),
            composition=[FluorophoreComponent.from_dict(c) for c in data.get('composition', [])],
            normalize=data.get('normalize', True)
        )
    
    def get_composition_for_object(self) -> List[Dict[str, Any]]:
        """Get composition in format expected by spatial.py"""
        return [c.to_dict() for c in self.composition]


class TemplateManager:
    """Manages object templates."""
    
    def __init__(self):
        self.templates: List[ObjectTemplate] = []
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create built-in default templates."""
        # Single fluorophore templates (for backward compatibility)
        for i in range(5):  # Support up to 5 fluorophores by default
            self.templates.append(ObjectTemplate(
                name=f"F{i+1} Only",
                description=f"Pure fluorophore {i+1}",
                composition=[FluorophoreComponent(fluor_index=i, ratio=1.0, ratio_noise=0.0)],
                normalize=False
            ))
        
        # Co-localization templates
        self.templates.append(ObjectTemplate(
            name="F1+F2 Equal",
            description="Equal mix of F1 and F2",
            composition=[
                FluorophoreComponent(fluor_index=0, ratio=0.5, ratio_noise=0.05),
                FluorophoreComponent(fluor_index=1, ratio=0.5, ratio_noise=0.05)
            ]
        ))
        
        self.templates.append(ObjectTemplate(
            name="F1+F2 Dominant F1",
            description="F1-dominant with some F2",
            composition=[
                FluorophoreComponent(fluor_index=0, ratio=0.7, ratio_noise=0.1),
                FluorophoreComponent(fluor_index=1, ratio=0.3, ratio_noise=0.05)
            ]
        ))
        
        self.templates.append(ObjectTemplate(
            name="Triple Label",
            description="Three fluorophores with variable ratios",
            composition=[
                FluorophoreComponent(fluor_index=0, ratio=0.5, ratio_noise=0.1),
                FluorophoreComponent(fluor_index=1, ratio=0.3, ratio_noise=0.08),
                FluorophoreComponent(fluor_index=2, ratio=0.2, ratio_noise=0.05)
            ]
        ))
        
        self.templates.append(ObjectTemplate(
            name="Contaminated",
            description="Main fluorophore with small contamination",
            composition=[
                FluorophoreComponent(fluor_index=0, ratio=0.85, ratio_noise=0.05),
                FluorophoreComponent(fluor_index=1, ratio=0.15, ratio_noise=0.03)
            ]
        ))
    
    def add_template(self, template: ObjectTemplate):
        """Add a new template."""
        self.templates.append(template)
    
    def remove_template(self, name: str):
        """Remove a template by name."""
        self.templates = [t for t in self.templates if t.name != name]
    
    def get_template(self, name: str) -> ObjectTemplate | None:
        """Get a template by name."""
        for t in self.templates:
            if t.name == name:
                return t
        return None
    
    def get_template_names(self) -> List[str]:
        """Get list of all template names."""
        return [t.name for t in self.templates]
    
    def save_to_file(self, filepath: str):
        """Save templates to JSON file."""
        data = {
            'templates': [t.to_dict() for t in self.templates if not t.name.endswith(' Only')]  # Skip default single-fluor templates
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load templates from JSON file."""
        if not Path(filepath).exists():
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Add loaded templates (don't replace defaults)
        for t_data in data.get('templates', []):
            template = ObjectTemplate.from_dict(t_data)
            # Remove existing template with same name if exists
            self.remove_template(template.name)
            self.add_template(template)


