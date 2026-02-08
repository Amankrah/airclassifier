"""
Recipe Management
=================

Mirrors the GP-15 HMI recipe system: up to 30 stored recipes,
each defining electrode gap, belt speed, fan/heater settings,
anode current limits, and optional temperature control.
"""

from __future__ import annotations

from typing import Dict, Optional, List

from ..config import Recipe


class RecipeStore:
    """In-memory mirror of the GP-15's 30-recipe storage.

    Recipes are indexed 1-30. Recipe 0 = manual mode.
    """

    def __init__(self, capacity: int = 30):
        self._capacity = capacity
        self._recipes: Dict[int, Recipe] = {}

    def store(self, recipe: Recipe) -> None:
        """Store a recipe at its recipe_number slot."""
        if recipe.recipe_number < 0 or recipe.recipe_number > self._capacity:
            raise ValueError(f"Recipe number must be 0-{self._capacity}")
        self._recipes[recipe.recipe_number] = recipe

    def load(self, recipe_number: int) -> Optional[Recipe]:
        """Load a recipe by number. Returns None if slot is empty."""
        return self._recipes.get(recipe_number)

    def list_recipes(self) -> List[Recipe]:
        """Return all stored recipes."""
        return list(self._recipes.values())

    def clear(self, recipe_number: int) -> None:
        """Clear a recipe slot."""
        self._recipes.pop(recipe_number, None)
