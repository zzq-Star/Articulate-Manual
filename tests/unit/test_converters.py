"""Tests for deployment script converters."""

import tempfile
from pathlib import Path

import pytest

from articulate_core.skill.converters.base import Trajectory
from articulate_core.skill.converters.factory import ConverterFactory
from articulate_core.skill.converters.ur_script import URScriptConverter
from articulate_core.skill.converters.krl import KRLConverter
from articulate_core.skill.converters.rapid import RAPIDConverter


@pytest.fixture
def sample_trajectory():
    return Trajectory(
        waypoints=[
            {"positions": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0], "type": "PTP"},
            {"positions": [0.3, 0.0, 0.3, 0.0, 0.0, 0.0], "type": "LIN"},
            {"positions": [0.3, 0.3, 0.3, 0.0, 0.0, 0.0], "type": "PTP"},
        ],
        speed=0.25,
        blend=0.0,
        tool_name="tool0",
        payload_kg=1.0,
    )


class TestURScriptConverter:
    def test_brand(self):
        assert URScriptConverter().brand == "ur"

    def test_convert_returns_script(self, sample_trajectory):
        converter = URScriptConverter()
        with tempfile.TemporaryDirectory() as tmpdir:
            files = converter.convert(sample_trajectory, tmpdir)
            assert "articulate_program.script" in files
            content = files["articulate_program.script"]
            assert "def articulate_program():" in content
            assert "movej" in content
            assert "movel" in content
            assert "end" in content.lower()

    def test_safety_checklist(self):
        converter = URScriptConverter()
        items = converter.generate_safety_checklist()
        assert len(items) > 0
        assert any("TCP" in item for item in items)

    def test_deployment_guide(self, sample_trajectory):
        converter = URScriptConverter()
        guide = converter.generate_deployment_guide(sample_trajectory)
        assert "UR" in guide.upper()
        assert "Waypoints" in guide


class TestKRLConverter:
    def test_brand(self):
        assert KRLConverter().brand == "kuka"

    def test_convert_returns_src_and_dat(self, sample_trajectory):
        converter = KRLConverter()
        with tempfile.TemporaryDirectory() as tmpdir:
            files = converter.convert(sample_trajectory, tmpdir)
            assert "articulate_program.src" in files
            assert "articulate_program.dat" in files
            src = files["articulate_program.src"]
            dat = files["articulate_program.dat"]
            assert "DEF articulate_program" in src
            assert "END" in src.splitlines()[-1]
            assert "DEFDAT" in dat

    def test_safety_checklist(self):
        converter = KRLConverter()
        items = converter.generate_safety_checklist()
        assert len(items) > 0
        assert any("KUKA" in item for item in items)


class TestRAPIDConverter:
    def test_brand(self):
        assert RAPIDConverter().brand == "abb"

    def test_convert_returns_mod(self, sample_trajectory):
        converter = RAPIDConverter()
        with tempfile.TemporaryDirectory() as tmpdir:
            files = converter.convert(sample_trajectory, tmpdir)
            assert "articulate_program.mod" in files
            content = files["articulate_program.mod"]
            assert "MODULE" in content
            assert "ENDMODULE" in content
            assert "MoveJ" in content
            assert "MoveL" in content

    def test_safety_checklist(self):
        converter = RAPIDConverter()
        items = converter.generate_safety_checklist()
        assert len(items) > 0
        assert any("TCP" in item for item in items)


class TestConverterFactory:
    def test_get_converter(self):
        conv = ConverterFactory.get_converter("ur")
        assert isinstance(conv, URScriptConverter)
        conv = ConverterFactory.get_converter("kuka")
        assert isinstance(conv, KRLConverter)
        conv = ConverterFactory.get_converter("abb")
        assert isinstance(conv, RAPIDConverter)

    def test_get_converter_invalid(self):
        with pytest.raises(ValueError, match="Unsupported brand"):
            ConverterFactory.get_converter("nonexistent")

    def test_list_brands(self):
        brands = ConverterFactory.list_brands()
        assert "ur" in brands
        assert "kuka" in brands
        assert "abb" in brands

    def test_list_descriptions(self):
        descs = ConverterFactory.list_descriptions()
        brands = {d["brand"] for d in descs}
        assert brands == {"ur", "kuka", "abb"}
