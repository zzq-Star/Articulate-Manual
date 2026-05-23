"""Converter factory — maps brand identifiers to converter instances."""

from typing import Dict, Optional

from articulate_core.skill.converters.base import BaseConverter
from articulate_core.skill.converters.ur_script import URScriptConverter
from articulate_core.skill.converters.krl import KRLConverter
from articulate_core.skill.converters.rapid import RAPIDConverter


class ConverterFactory:
    """Registry and factory for brand-specific converters.

    Usage:
        converter = ConverterFactory.get_converter("ur")
        files = converter.convert(trajectory, "/output/dir")
    """

    _registry: Dict[str, type[BaseConverter]] = {}

    @classmethod
    def register(cls, brand: str, converter_cls: type[BaseConverter]):
        """Register a converter class for a brand."""
        cls._registry[brand] = converter_cls

    @classmethod
    def get_converter(cls, brand: str) -> BaseConverter:
        """Get converter instance for given brand.

        Raises:
            ValueError: If no converter is registered for the brand.
        """
        if brand not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unsupported brand: {brand!r}. "
                f"Available brands: {available}"
            )
        return cls._registry[brand]()

    @classmethod
    def list_brands(cls) -> list[str]:
        """Return sorted list of registered brand identifiers."""
        return sorted(cls._registry)

    @classmethod
    def list_descriptions(cls) -> list[dict]:
        """Return list of brand info dicts."""
        results = []
        for brand, cls_type in sorted(cls._registry.items()):
            inst = cls_type()
            results.append({
                "brand": brand,
                "description": inst.__class__.__doc__ or "",
                "file_extension": cls_type._file_extension()
                if hasattr(cls_type, "_file_extension")
                else ".script",
            })
        return results


# Auto-register built-in converters
ConverterFactory.register("ur", URScriptConverter)
ConverterFactory.register("kuka", KRLConverter)
ConverterFactory.register("abb", RAPIDConverter)
