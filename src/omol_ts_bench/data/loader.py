"""Load reaction data from XYZ files and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms
from ase.io import read


@dataclass
class Reaction:
    """A single reaction with reactant and product endpoints."""

    reaction_id: str
    reactant: Atoms
    product: Atoms
    tier: str = "full"
    metadata: dict = field(default_factory=dict)


def load_reaction(
    reactant_path: str | Path,
    product_path: str | Path,
    reaction_id: str | None = None,
) -> Reaction:
    """Load a single reaction from XYZ files.

    Parameters
    ----------
    reactant_path
        Path to reactant XYZ file.
    product_path
        Path to product XYZ file.
    reaction_id
        Optional reaction ID. If not provided, derived from filename.

    Returns
    -------
    Reaction
        Loaded reaction object.
    """
    reactant_path = Path(reactant_path)
    product_path = Path(product_path)

    reactant = read(str(reactant_path))
    product = read(str(product_path))

    if reaction_id is None:
        reaction_id = reactant_path.stem

    return Reaction(
        reaction_id=reaction_id,
        reactant=reactant,
        product=product,
    )


def load_reactions_from_directory(
    data_dir: str | Path,
    tier: str | None = None,
    registry_path: str | Path | None = None,
) -> list[Reaction]:
    """Load all reactions from a data directory.

    Expects the following structure:
        data_dir/
            reactants/
                rxn_0001.xyz
                rxn_0002.xyz
                ...
            products/
                rxn_0001.xyz
                rxn_0002.xyz
                ...

    Parameters
    ----------
    data_dir
        Root data directory.
    tier
        If provided, only load reactions in this tier ("smoke", "standard", "full").
        Requires registry_path.
    registry_path
        Path to reactions.yaml registry file for tier filtering.

    Returns
    -------
    list[Reaction]
        List of loaded reactions.
    """
    import yaml

    data_dir = Path(data_dir)
    reactant_dir = data_dir / "xyz" / "reactants"
    product_dir = data_dir / "xyz" / "products"

    # Load tier info if filtering
    tier_map: dict[str, str] = {}
    if registry_path is not None:
        registry_path = Path(registry_path)
        with open(registry_path) as f:
            registry = yaml.safe_load(f)
        for rxn_entry in registry.get("reactions", []):
            tier_map[rxn_entry["id"]] = rxn_entry.get("tier", "full")

    reactions = []
    for rxn_file in sorted(reactant_dir.glob("*.xyz")):
        rxn_id = rxn_file.stem
        product_file = product_dir / rxn_file.name

        if not product_file.exists():
            continue

        # Apply tier filter
        rxn_tier = tier_map.get(rxn_id, "full")
        if tier is not None and rxn_tier != tier:
            # Include if the requested tier is less restrictive
            tier_order = {"smoke": 0, "standard": 1, "full": 2}
            if tier_order.get(rxn_tier, 2) > tier_order.get(tier, 2):
                continue

        rxn = load_reaction(rxn_file, product_file, reaction_id=rxn_id)
        rxn.tier = rxn_tier
        reactions.append(rxn)

    return reactions
