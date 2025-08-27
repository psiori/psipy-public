# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Authors: Sascha Lange <sascha.lange@psiori.com>

import re
from datetime import datetime
import tempfile
from unittest.mock import patch

import pytest

from psipy.core.id_mixin import IDMixin
from psipy.core.io import Saveable


class TestIDClass(IDMixin):
    """Simple test class that uses IDMixin for testing purposes."""
    
    def __init__(self, name="test", **kwargs):
        self.name = name
        super().__init__(**kwargs)

class TestIDSaveableClass(IDMixin, Saveable):
    """Simple test class that uses IDMixin for testing purposes."""
    
    def __init__(self, name="test", **kwargs):
        self.name = name
        super().__init__(**kwargs)  # pass the 
        self.update_config(name=name, id=self.id)


class TestIDMixin:
    """Test suite for IDMixin functionality."""

    def test_id_generation_on_init(self):
        """Test that ID is automatically generated during initialization."""
        obj = TestIDClass()
        
        # Check that ID components are generated
        assert obj.id_strand is not None
        assert obj.id_generation == 0
        assert obj.id is not None
        
        # Check strand format (YYYYMMDDHHMMSS-XXXXX)
        strand_pattern = r'^20\d{12}-[0-9A-F]{5}$'
        assert re.match(strand_pattern, obj.id_strand), f"Invalid strand format: {obj.id_strand}"
        
        # Check full ID format
        full_id_pattern = r'^20\d{12}-[0-9A-F]{5}\_\d{4}$'
        assert re.match(full_id_pattern, obj.id), f"Invalid full ID format: {obj.id}"

    def test_strand_format(self):
        """Test that strand follows the correct format."""
        obj = TestIDClass()
        strand = obj.id_strand
        
        # Should be YYYYMMDDHHMMSS-XXXXX format
        parts = strand.split('-')
        assert len(parts) == 2, f"Strand should have exactly one hyphen: {strand}"
        
        timestamp_part, hex_part = parts
        
        # Check timestamp part (YYYYMMDDHHMMSS)
        assert len(timestamp_part) == 14, f"Timestamp should be 14 digits: {timestamp_part}"
        assert timestamp_part.isdigit(), f"Timestamp should be all digits: {timestamp_part}"
        assert timestamp_part.startswith('20'), f"Year should start with 20: {timestamp_part}"
        
        # Check hex part (5 uppercase hex characters)
        assert len(hex_part) == 5, f"Hex part should be 5 characters: {hex_part}"
        assert all(c in '0123456789ABCDEF' for c in hex_part), f"Hex part should be uppercase hex: {hex_part}"

    def test_generation_starts_at_zero(self):
        """Test that generation starts at 0."""
        obj = TestIDClass()
        assert obj.id_generation == 0
        assert obj.id.endswith('_0000')

    def test_increment_generation_default(self):
        """Test generation increment with default value (1)."""
        obj = TestIDClass()
        original_strand = obj.id_strand
        
        obj.increment_generation()
        
        assert obj.id_generation == 1
        assert obj.id_strand == original_strand  # Strand should not change
        assert obj.id.endswith('_0001')

    def test_increment_generation_custom(self):
        """Test generation increment with custom value."""
        obj = TestIDClass()
        original_strand = obj.id_strand
        
        obj.increment_generation(5)
        
        assert obj.id_generation == 5
        assert obj.id_strand == original_strand  # Strand should not change
        assert obj.id.endswith('_0005')

    def test_increment_generation_multiple(self):
        """Test multiple generation increments."""
        obj = TestIDClass()
        original_strand = obj.id_strand
        
        # Test multiple increments
        for i in range(1, 10):
            obj.increment_generation()
            assert obj.id_generation == i
            assert obj.id_strand == original_strand  # Strand should not change
            expected_gen = f"{i:04d}"
            assert obj.id.endswith(f"_{expected_gen}")

    def test_large_generation_numbers(self):
        """Test generation numbers larger than 9999."""
        obj = TestIDClass()
        
        # Test large generation (> 9999)
        obj.increment_generation(12345)
        assert obj.id_generation == 12345
        assert obj.id.endswith('_12345')  # Should not be zero-padded

    def test_set_id_from_string_valid(self):
        """Test setting ID from valid string format."""
        obj = TestIDClass()
        
        test_id = "20241201143022-A3F2E_0005"
        obj.set_id_from_string(test_id)
        
        assert obj.id_strand == "20241201143022-A3F2E"
        assert obj.id_generation == 5
        assert obj.id == test_id

    def test_set_id_from_string_large_generation(self):
        """Test setting ID with large generation number."""
        obj = TestIDClass()
        
        test_id = "20241201143022-A3F2E_12345"
        obj.set_id_from_string(test_id)
        
        assert obj.id_strand == "20241201143022-A3F2E"
        assert obj.id_generation == 12345
        assert obj.id == test_id

    def test_set_id_from_string_invalid_formats(self):
        """Test that invalid ID strings raise ValueError."""
        obj = TestIDClass()
        
        invalid_ids = [
            "invalid",
            "20241201143022_0005",  # Missing hex part
            "20241201143022-a3f2e_0005",  # Lowercase hex
            "20241201143022-A3F2E_-1",  # Negative generation
            "20241201143022-A3F2E_abc",  # Non-numeric generation
            "19991201143022-A3F2E_0005",  # Invalid year (before 2000)
            "20241201143022-A3F2E",  # Missing generation
            "20241201143022-A3F2E_0005_extra",  # Extra parts
            "20241201143022-A3F2E_",  # Empty generation
            "20241201143022-A3F2_0005",  # Hex part too short
            "20241201143022-A3F2EF_0005",  # Hex part too long
            "2024120114302-A3F2E_0005",  # Timestamp too short
            "202412011430223-A3F2E_0005",  # Timestamp too long
        ]
        
        for invalid_id in invalid_ids:
            with pytest.raises(ValueError):
                obj.set_id_from_string(invalid_id)

    def test_set_id_from_string_non_string(self):
        """Test that non-string input raises ValueError."""
        obj = TestIDClass()
        
        with pytest.raises(ValueError, match="ID must be a string"):
            obj.set_id_from_string(123)
        
        with pytest.raises(ValueError, match="ID must be a string"):
            obj.set_id_from_string(None)

    def test_from_string_class_method(self):
        """Test creating instance from ID string using class method."""
        test_id = "20241201143022-A3F2E_0007"
        obj = TestIDClass.from_string(test_id)
        
        assert isinstance(obj, TestIDClass)
        assert obj.id == test_id
        assert obj.id_strand == "20241201143022-A3F2E"
        assert obj.id_generation == 7

    def test_from_string_invalid(self):
        """Test that from_string raises ValueError for invalid ID."""
        with pytest.raises(ValueError):
            TestIDClass.from_string("invalid_id")

    def test_unique_ids_across_instances(self):
        """Test that different instances get unique IDs."""
        obj1 = TestIDClass()
        obj2 = TestIDClass()
        
        # IDs should be different
        assert obj1.id != obj2.id
        assert obj1.id_strand != obj2.id_strand
        # Both should start at generation 0
        assert obj1.id_generation == 0
        assert obj2.id_generation == 0

    def test_id_property_formatting(self):
        """Test ID property formatting for various generation values."""
        obj = TestIDClass()
        original_strand = obj.id_strand
        
        # Test 4-digit padding for small numbers
        test_cases = [
            (0, "_0000"),
            (1, "_0001"), 
            (99, "_0099"),
            (999, "_0999"),
            (9999, "_9999"),
            (10000, "_10000"),  # No padding for > 9999
            (123456, "_123456"),
        ]
        
        for generation, expected_suffix in test_cases:
            obj._generation = generation
            assert obj.id == f"{original_strand}{expected_suffix}"

    @patch('psipy.core.id_mixin.datetime')
    def test_deterministic_timestamp(self, mock_datetime):
        """Test that strand generation uses current datetime."""
        # Mock datetime to return a fixed time
        mock_datetime.now.return_value.strftime.return_value = "20241201143022"
        
        with patch('psipy.core.id_mixin.secrets.choice') as mock_choice:
            # Mock random hex generation
            mock_choice.side_effect = ['A', 'B', 'C', 'D', 'E']
            
            obj = TestIDClass()
            
            assert obj.id_strand == "20241201143022-ABCDE"
            mock_datetime.now.assert_called_once()
            mock_datetime.now.return_value.strftime.assert_called_once_with("%Y%m%d%H%M%S")

    def test_inheritance_compatibility(self):
        """Test that IDMixin works correctly with inheritance."""
        class ComplexTestClass(IDMixin):
            def __init__(self, value, **kwargs):
                self.value = value
                super().__init__(**kwargs)
                
            def get_value(self):
                return self.value
        
        obj = ComplexTestClass("test_value")
        
        # Should have ID functionality
        assert hasattr(obj, 'id')
        assert hasattr(obj, 'id_strand')
        assert hasattr(obj, 'id_generation')
        assert hasattr(obj, 'increment_generation')
        
        # Should have its own functionality
        assert obj.get_value() == "test_value"
        
        # ID should be properly generated
        assert re.match(r'^20\d{12}-[0-9A-F]{5}\_\d{4}$', obj.id)

    def test_regression_generation_validation(self):
        """Test edge cases and regression scenarios."""
        obj = TestIDClass()
        
        # Test setting very large generation numbers
        obj._generation = 999999
        assert obj.id.endswith('_999999')
        
        # Test zero generation after increment
        obj._generation = 0
        obj.increment_generation(0)  # Increment by 0
        assert obj.id_generation == 0
        
        # Test negative increment (should be possible but unusual)
        obj._generation = 10
        obj.increment_generation(-5)
        assert obj.id_generation == 5

    def test_properties_read_only_behavior(self):
        """Test that ID properties behave correctly."""
        obj = TestIDClass()
        
        # Properties should return consistent values
        strand1 = obj.id_strand
        strand2 = obj.id_strand
        assert strand1 == strand2
        
        generation1 = obj.id_generation
        generation2 = obj.id_generation
        assert generation1 == generation2
        
        id1 = obj.id
        id2 = obj.id
        assert id1 == id2
        
        # After increment, properties should reflect changes
        obj.increment_generation()
        assert obj.id_generation == generation1 + 1
        assert obj.id != id1  # ID should change
        assert obj.id_strand == strand1  # Strand should not change


    def test_save_load_behavior(self):
        """Test saving and loading preserves ID information. 
        
        We test this here, as the there was a bug due to an incompatibility of the IDMixin with the (older) Saveable class."""
        obj = TestIDSaveableClass()
        
        # Get initial state
        original_id = obj.id
        original_strand = obj.id_strand
        original_generation = obj.id_generation
        
        # Save to memory
        with tempfile.NamedTemporaryFile() as tmp:
            obj.save(tmp.name)
            print(f"Saved to {tmp.name}")

            # Load into new object
            loaded_obj = TestIDSaveableClass.load(tmp.name + ".zip")
            
            # Check ID components match
            assert loaded_obj.id == original_id
            assert loaded_obj.id_strand == original_strand
            assert loaded_obj.id_generation == original_generation
            
            # Verify loaded object maintains functionality
            assert loaded_obj.name == "test"
            
            # Verify generation increment still works
            loaded_obj.increment_generation()
            assert loaded_obj.id_generation == original_generation + 1
            assert loaded_obj.id_strand == original_strand
