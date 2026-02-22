"""
Milling Recipe Management
=========================

Recipe storage and management for hammer mill operating setpoints.
Mirrors the HMI recipe system used in the GP-15 pretreatment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..config import MillRecipe


@dataclass
class RecipeStore:
    """Storage for milling recipes.

    Supports up to 30 recipes, matching industrial practice.
    """

    recipes: Dict[int, MillRecipe] = None
    max_recipes: int = 30

    def __post_init__(self):
        if self.recipes is None:
            self.recipes = {}

    def save_recipe(self, recipe: MillRecipe) -> int:
        """Save a recipe to the store.

        Args:
            recipe: Recipe to save

        Returns:
            Recipe number (slot index)
        """
        num = recipe.recipe_number
        if num < 1 or num > self.max_recipes:
            # Find next available slot
            for i in range(1, self.max_recipes + 1):
                if i not in self.recipes:
                    num = i
                    break
            else:
                raise ValueError("Recipe store is full")

        self.recipes[num] = recipe
        return num

    def load_recipe(self, recipe_number: int) -> Optional[MillRecipe]:
        """Load a recipe from the store.

        Args:
            recipe_number: Recipe slot number (1-30)

        Returns:
            Recipe if found, None otherwise
        """
        return self.recipes.get(recipe_number)

    def delete_recipe(self, recipe_number: int) -> bool:
        """Delete a recipe from the store.

        Args:
            recipe_number: Recipe slot number

        Returns:
            True if deleted, False if not found
        """
        if recipe_number in self.recipes:
            del self.recipes[recipe_number]
            return True
        return False

    def list_recipes(self) -> List[MillRecipe]:
        """List all stored recipes.

        Returns:
            List of recipes sorted by number
        """
        return [self.recipes[k] for k in sorted(self.recipes.keys())]

    def to_json(self, path: Path):
        """Save recipes to JSON file.

        Args:
            path: Output file path
        """
        data = {
            str(k): asdict(v)
            for k, v in self.recipes.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "RecipeStore":
        """Load recipes from JSON file.

        Args:
            path: Input file path

        Returns:
            RecipeStore with loaded recipes
        """
        with open(path, "r") as f:
            data = json.load(f)

        store = cls()
        for k, v in data.items():
            recipe = MillRecipe(**v)
            recipe.recipe_number = int(k)
            store.recipes[int(k)] = recipe

        return store


# Default recipes for common operations
DEFAULT_RECIPES = {
    1: MillRecipe(
        name="Fine Flour",
        recipe_number=1,
        rotor_rpm=3500,
        screen_aperture_mm=1.0,
        feed_rate_kg_per_hr=400,
    ),
    2: MillRecipe(
        name="Medium Flour",
        recipe_number=2,
        rotor_rpm=3000,
        screen_aperture_mm=1.5,
        feed_rate_kg_per_hr=500,
    ),
    3: MillRecipe(
        name="Coarse Flour",
        recipe_number=3,
        rotor_rpm=2500,
        screen_aperture_mm=2.0,
        feed_rate_kg_per_hr=600,
    ),
}
