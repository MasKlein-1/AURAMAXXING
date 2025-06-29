from abc import ABC, abstractmethod
from ase import Atoms
from pathlib import Path
import os
from helpers.files_io import add_dump_to_traj
from interfaces.MACE_interface import MACEInterface
from interfaces.LAMMPS_Interface import LMPInterface
from ase.constraints import FixAtoms

class BaseOptimizer(ABC):
    """Abstract base class for optimizers"""
    @abstractmethod
    def optimize(self, atoms: Atoms, opt_type: str, **kwargs) -> Atoms:
        pass
    @staticmethod
    def process_dump(traj_file: str, dump_path: Path):
        if dump_path.exists():
            add_dump_to_traj(dump_path, traj_file)
            os.remove(dump_path)

class LammpsOptimizer(BaseOptimizer):
    """Wrapper for LAMMPS optimizer/annealer"""
    def __init__(self, struc):
        self.optimizer = LMPInterface()
        self.dump_path = Path("LAMMPS") / "dump.xyz"

    def optimize(self, atoms: Atoms, opt_type: str, **kwargs) -> Atoms:
        if opt_type == "anneal":
            frozen_atoms = (
                kwargs["frozen_indices"] if "frozen_indices" in kwargs else []
            )
            return self.optimizer.set_task(
                atoms=atoms,
                type_opt="anneal",
                n_steps_heating=kwargs.get("n_steps_heating", 1000),
                n_steps_cooling=kwargs.get("n_steps_cooling", 1000),
                start_T=kwargs["start_T"],
                final_T=kwargs["final_T"],
                frozen_atoms=frozen_atoms,
            )
        elif opt_type == "final":
            return self.optimizer.set_task(
                "final",
                steps=kwargs["steps"],
                start_T=kwargs["start_T"],
                final_T=kwargs["final_T"],
                FF="BKS",
            )
        elif opt_type == "minimize":
            max_steps = kwargs["max_steps"] if "max_steps" in kwargs else 500
            frozen_atoms = (
                kwargs["frozen_indices"] if "frozen_indices" in kwargs.keys() else None
            )
            minimized = self.optimizer.set_task(
                type_opt="minimize",
                atoms=atoms,
                max_steps=max_steps,
                frozen_atoms=frozen_atoms,
            )
            return minimized


class MACEOptimizer(BaseOptimizer):
    "Wrapper for MACE optimizer/annealer"
    def __init__(self, model_path: str, device="mps"):
        self.optimizer = MACEInterface(model_path, device=device)
        self.dump_path = Path("dump.xyz")

    def optimize(self, atoms: Atoms, opt_type: str, **kwargs) -> Atoms:
        if opt_type == "anneal":
            frozen_atoms = (
                kwargs["frozen_indices"] if "frozen_indices" in kwargs.keys() else None
            )
            return self.optimizer.set_task(
                atoms=atoms,
                type_opt="anneal",
                n_steps_heating=kwargs.get("n_steps_heating", 1000),
                n_steps_cooling=kwargs.get("n_steps_cooling", 1000),
                start_T=kwargs["start_T"],
                final_T=kwargs["final_T"],
                frozen_atoms=frozen_atoms,
            )
        elif opt_type == "minimize":
            max_steps = kwargs["max_steps"] if "max_steps" in kwargs.keys() else 500
            frozen_atoms = (
                kwargs["frozen_indices"] if "frozen_indices" in kwargs.keys() else None
            )
            minimized = self.optimizer.set_task(
                type_opt="minimize",
                atoms=atoms,
                max_steps=max_steps,
                frozen_atoms=frozen_atoms,
            )
            return minimized
