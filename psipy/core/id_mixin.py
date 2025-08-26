# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Authors: Sascha Lange <sascha.lange@psiori.com>

"""ID Mixin for providing unique identifiers with strand and generation.
================================================================

The ID consists of two parts:
- strand: DateTime (YYYYMMDDHHMMSS) + 5 random HEX characters, separated by hyphen
- generation: Non-negative integer (zero-padded to 4 digits), counting updates

Full ID format: strand_generation (e.g., "20241201143022-A3F2E_0001")
"""

import re
import secrets
from datetime import datetime
from typing import Dict, Any, Optional


class IDMixin:
    """Mixin class providing unique ID functionality with strand and generation.
    
    The ID has two components:
    - strand: Timestamp + random hex suffix (e.g., "20241201143022-A3F2E")
    - generation: Zero-padded integer counting updates (e.g., "0001")
    
    Full ID format: "{strand}_{generation}"
    """
    
    def __init__(self, id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if id is None:
            self._generate_new_id()
        else:
            self.set_id_from_string(id)
    
    def _generate_new_id(self) -> None:
        """Generate a new ID with fresh strand and generation 0."""
        # Generate timestamp part (YYYYMMDDHHMMSS)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Generate 5 random hex characters (uppercase)
        random_hex = ''.join(secrets.choice('0123456789ABCDEF') for _ in range(5))
        
        # Combine with hyphen
        self._strand = f"{timestamp}-{random_hex}"
        self._generation = 0

    @classmethod
    def from_string(cls, id_string: str) -> "IDMixin":
        """Create a new instance from an ID string.
        
        Args:
            id_string: ID string in format "strand_generation"
            
        Returns:
            A new instance with the ID parsed from the string
            
        Raises:
            ValueError: If the ID string format is invalid
        """
        obj = cls()
        obj.set_id_from_string(id_string)
        return obj
    
    @property
    def id_strand(self) -> str:
        """Get the strand part of the ID."""
        return self._strand
    
    @property
    def id_generation(self) -> int:
        """Get the generation part of the ID."""
        return self._generation
    
    @property
    def id(self) -> str:
        """Get the complete ID in format strand_generation."""
        # Pad generation to 4 digits, but allow more for generations > 9999
        generation_str = f"{self._generation:04d}" if self._generation <= 9999 else str(self._generation)
        return f"{self._strand}_{generation_str}"
    
    def increment_generation(self, increment: int = 1) -> None:
        """Increment the generation counter by increment."""
        self._generation += increment
    
    def set_id_from_string(self, id_string: str) -> None:
        """Parse and set ID from string format.
        
        Args:
            id_string: ID string in format "strand_generation"
            
        Raises:
            ValueError: If the ID string format is invalid
        """
        if not isinstance(id_string, str):
            raise ValueError(f"ID must be a string, got {type(id_string)}")
        
        # Split on the last underscore to handle potential underscores in timestamp
        parts = id_string.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid ID format: '{id_string}'. Expected format: 'strand_generation'")
        
        strand, generation_str = parts
        
        # Validate strand format: YYYYMMDDHHMMSS-XXXXX
        strand_pattern = r'^20\d{12}-[0-9A-F]{5}$'
        if not re.match(strand_pattern, strand):
            raise ValueError(
                f"Invalid strand format: '{strand}'. Expected format: 'YYYYMMDDHHMMSS-XXXXX' "
                f"where XXXXX are uppercase hex characters"
            )
        
        # Validate and parse generation
        try:
            generation = int(generation_str)
            if generation < 0:
                raise ValueError(f"Generation must be non-negative, got {generation}")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid generation format: '{generation_str}'. Must be a non-negative integer")
            raise
        
        # If validation passes, set the values
        self._strand = strand
        self._generation = generation




__all__ = ["IDMixin"]
