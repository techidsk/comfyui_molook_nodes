#!/usr/bin/env python

"""Tests for `comfyui_molook_nodes` package."""

import pytest
from src.molook_nodes.llm import OpenAIProviderNode

@pytest.fixture
def openai_provider_node():
    """Fixture to create an OpenAIProviderNode instance."""
    return OpenAIProviderNode()

def test_openai_provider_node_initialization(openai_provider_node):
    """Test that the node can be instantiated."""
    assert isinstance(openai_provider_node, OpenAIProviderNode)

def test_return_types():
    """Test the node's metadata."""
    assert OpenAIProviderNode.RETURN_TYPES == ("OPENAI_PROVIDER",)
    assert OpenAIProviderNode.FUNCTION == "execute"
    assert OpenAIProviderNode.CATEGORY == "Molook_nodes/LLM"
