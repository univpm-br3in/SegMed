"""
nnU-Net V1 Handler for SegMed
Handles loading and inference with nnU-Net V1 models
"""

import SimpleITK as sitk
from nnunet.inference import predict
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import pickle

import torch.nn.functional as F
from monai.inferers import sliding_window_inference as monai_sliding_window
from nnunet.inference.predict import predict_from_folder
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.preprocessing.preprocessing import get_do_separate_z, get_lowres_axis, \
    resample_data_or_seg, get_case_identifier_from_npz
from nnunet.preprocessing.preprocessing import GenericPreprocessor
from batchgenerators.utilities.file_and_folder_operations import load_pickle, maybe_mkdir_p

# CRITICAL FIX: PyTorch 2.6 changed torch.load default to weights_only=True
# This breaks nnU-Net V1 which has numpy dtypes in checkpoints
# Monkey-patch torch.load to always use weights_only=False
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for compatibility"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# Fix for Windows multiprocessing with nnU-Net V1
# nnU-Net V1 uses lambda functions that can't be pickled on Windows
# Monkey-patch nnunet to disable multiprocessing preprocessing

# Save original function
_original_preprocess = predict.preprocess_multithreaded

# Replace with synchronous version


def preprocess_sync(trainer, list_of_lists, output_files, num_processes=1,
                    pin_memory=False):
    """Synchronous preprocessing to avoid Windows pickling issues

    Args:
        trainer: nnUNetTrainerV2 object (contains the data loader and preprocessing)
        list_of_lists: List of input file lists
        output_files: List of output file paths
        num_processes: Number of processes (ignored, we use 1)
        pin_memory: Whether to pin memory (ignored)

    Yields:
        Tuple of (output_filename, (data, properties_dict))
    """
    # Process items synchronously
    for idx, (data_files, output_file) in enumerate(zip(list_of_lists, output_files)):
        # Use the trainer's preprocess_patient method
        # preprocess_patient in nnU-Net V1 returns: (data, None, properties)
        # where the middle element is for data augmentation (not used in inference)
        result = trainer.preprocess_patient(data_files)

        # Debug: Check what we actually got
        if isinstance(result, tuple):
            print(
                f"Debug: preprocess_patient returned tuple with {len(result)} elements")
            for i, elem in enumerate(result):
                if isinstance(elem, np.ndarray):
                    print(
                        f"  Element {i}: numpy array with shape {elem.shape}")
                elif isinstance(elem, dict):
                    print(
                        f"  Element {i}: dict with keys {list(elem.keys())[:5]}")
                else:
                    print(f"  Element {i}: {type(elem)}")

        # nnU-Net V1's preprocess_patient typically returns:
        # (data, None, properties_dict) for 3 elements
        # or (data, properties_dict) for 2 elements
        data = None
        properties = {}

        if isinstance(result, tuple):
            if len(result) == 3:
                # (data, None, properties) - standard V1 format
                data = result[0]
                properties = result[2]
            elif len(result) == 2:
                # (data, properties)
                data = result[0]
                properties = result[1]
            elif len(result) == 1:
                data = result[0]
            else:
                # Unknown format, try to handle it
                data = result[0]
                # Look for the dict in remaining elements
                for elem in result[1:]:
                    if isinstance(elem, dict):
                        properties = elem
                        break
        else:
            data = result

        # Ensure properties is a dictionary
        if not isinstance(properties, dict):
            if isinstance(properties, np.ndarray):
                print(
                    f"Warning: properties is a numpy array (shape {properties.shape}), this is unexpected!")
                print(
                    f"  This means we're unpacking incorrectly. Result has {len(result)} elements")
                properties = {}
            elif properties is None:
                properties = {}
            else:
                try:
                    properties = dict(properties)
                except:
                    print(
                        f"Warning: Could not convert properties of type {type(properties)} to dict")
                    properties = {}

        # Yield in the format expected by nnU-Net: (output_file, (data, properties))
        yield output_file, (data, properties)


# Apply the patch
predict.preprocess_multithreaded = preprocess_sync
# print("  ✓ Applied Windows multiprocessing fix for nnU-Net V1")


class NnUNetV1Handler:
    """Handler for nnU-Net V1 models."""

    def __init__(self):
        self.model = None
        self.model_folder = None
        self.plans = None
        self.dataset_json = None
        self.configuration = None  # '2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'
        self.folds = None
        self.trainer_class = None
        self.checkpoint_name = 'model_final_checkpoint.model'

    def is_nnunet_v1_folder(self, folder_path: str) -> bool:
        """
        Check if the folder is a valid nnU-Net V1 trained model folder.

        Supports multiple V1 structures:
        1. Standard training output:
           folder/
           ├── plans.pkl
           └── fold_X/
               └── model_final_checkpoint.model

        2. Model Zoo downloads (simplified):
           folder/
           ├── *.pkl (any pkl file)
           └── *.model (checkpoint file, may be directly in folder or subfolder)

        3. Single model files:
           folder/
           └── model_final_checkpoint.model (with or without plans)
        """
        folder = Path(folder_path)

        # Check for any .pkl files (plans, postprocessing, etc.)
        pkl_files = list(folder.glob("**/*.pkl"))

        # Check for any .model checkpoint files
        model_files = list(folder.glob("**/*.model"))
        # Some V1 models use .pth
        model_pth_files = list(folder.glob("**/model_*.pth"))

        # V1 detection criteria (relaxed):
        # Must have at least ONE of:
        # - A .pkl file AND a .model file
        # - A fold_X directory with checkpoint
        # - A .model file with specific V1 naming convention

        has_pkl = len(pkl_files) > 0
        has_model = len(model_files) > 0 or len(model_pth_files) > 0
        has_fold_structure = any(folder.glob("fold_*"))

        # Check for V1-specific checkpoint naming
        v1_checkpoint_names = [
            "model_final_checkpoint.model",
            "model_best.model",
            "model_final_checkpoint.model.pkl"
        ]
        has_v1_checkpoint = any(
            (folder / name).exists() or
            any(folder.glob(f"*/{name}")) or
            any(folder.glob(f"*/all/{name}"))
            for name in v1_checkpoint_names
        )

        # Definite V1 indicators:
        if has_pkl and has_model:
            return True
        if has_fold_structure and has_model:
            return True
        if has_v1_checkpoint:
            return True

        # Check for "nnUNetTrainer" in any file/folder names (V1 convention)
        if has_model and any("nnUNetTrainer" in str(f) for f in folder.rglob("*")):
            return True

        return False

    def load_nnunet_v1_model(self, folder_path: str, device: torch.device,
                             folds: Optional[Tuple[int, ...]] = None,
                             checkpoint_name: str = 'model_final_checkpoint.model') -> bool:
        """
        Load nnU-Net V1 model with flexible folder structure support.

        Args:
            folder_path: Path to nnU-Net V1 trained model folder
            device: torch device (cpu or cuda)
            folds: Tuple of folds to use (e.g., (0,) or (0, 1, 2, 3, 4))
            checkpoint_name: Name of checkpoint to load

        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n=== Loading nnU-Net V1 Model ===")
            print(f"Model path: {folder_path}")

            self.model_folder = Path(folder_path)
            self.checkpoint_name = checkpoint_name

            # Step 1: Find plans file (flexible search)
            plans_pkl = self._find_plans_file()
            if plans_pkl:
                with open(plans_pkl, 'rb') as f:
                    self.plans = pickle.load(f)
                print(f"✓ Plans loaded from {plans_pkl.name}")
            else:
                print(f"⚠ No plans.pkl found (will attempt to proceed)")
                self.plans = None

            # Step 2: Load dataset.json if available (optional)
            dataset_json_path = self._find_dataset_json()
            if dataset_json_path:
                with open(dataset_json_path, 'r') as f:
                    self.dataset_json = json.load(f)
                print(f"✓ Dataset JSON loaded")
                print(f"  Dataset: {self.dataset_json.get('name', 'Unknown')}")
                print(f"  Labels: {self.dataset_json.get('labels', {})}")
            else:
                print(f"⚠ Dataset JSON not found (optional)")
                self.dataset_json = None

            # Step 3: Auto-detect configuration
            self.configuration = self._auto_detect_configuration()
            print(f"  Configuration: {self.configuration}")

            # Step 4: Find and load model checkpoints
            checkpoint_files = self._find_checkpoint_files()

            if not checkpoint_files:
                raise ValueError("No valid V1 checkpoint files found")

            print(f"  Found {len(checkpoint_files)} checkpoint(s)")

            # Step 5: Load the model(s)
            self.models = []
            for idx, checkpoint_path in enumerate(checkpoint_files):
                print(f"  Loading checkpoint {idx}: {checkpoint_path.name}...")

                try:
                    # Load checkpoint
                    checkpoint = torch.load(
                        str(checkpoint_path), map_location=device, weights_only=False)

                    # Extract the model state dict
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        # Checkpoint is the state dict itself
                        state_dict = checkpoint

                    # Store the state dict
                    self.models.append({
                        'fold': idx,
                        'state_dict': state_dict,
                        'device': device,
                        'path': checkpoint_path
                    })

                    print(f"  ✓ Checkpoint {idx} loaded")

                except Exception as e:
                    print(f"  ⚠ Failed to load checkpoint {idx}: {e}")
                    continue

            if not self.models:
                raise ValueError("No valid checkpoints could be loaded")

            print(f"✓ nnU-Net V1 model loaded successfully")
            print(f"  Number of checkpoints: {len(self.models)}")
            print(f"  Number of classes: {self.get_num_classes()}")
            print("=== nnU-Net V1 Model Ready ===\n")

            return True

        except Exception as e:
            print(f"✗ Error loading nnU-Net V1 model: {e}")
            return False

    def _find_plans_file(self) -> Optional[Path]:
        """Find plans.pkl file with flexible search."""
        # Try common locations
        candidates = [
            self.model_folder / "plans.pkl",
            self.model_folder / "nnUNetPlansv2.1_plans_3D.pkl",
            self.model_folder / "nnUNetPlans.pkl",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Search recursively for any .pkl file with "plan" in name
        plan_files = list(self.model_folder.glob("**/plans*.pkl"))
        if plan_files:
            return plan_files[0]

        # Search for any .pkl file
        pkl_files = list(self.model_folder.glob("**/*.pkl"))
        if pkl_files:
            # Filter out postprocessing files
            for pkl in pkl_files:
                if 'postprocessing' not in pkl.name.lower():
                    return pkl

        return None

    def _find_dataset_json(self) -> Optional[Path]:
        """Find dataset.json file."""
        candidates = [
            self.model_folder / "dataset.json",
            self.model_folder / "dataset_properties.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Search recursively
        json_files = list(self.model_folder.glob("**/dataset*.json"))
        return json_files[0] if json_files else None

    def _find_checkpoint_files(self) -> list:
        """Find checkpoint files with flexible search."""
        checkpoints = []

        # Pattern 1: Standard fold structure
        fold_dirs = list(self.model_folder.glob("fold_*"))
        for fold_dir in fold_dirs:
            for checkpoint_name in ["model_final_checkpoint.model", "model_best.model"]:
                checkpoint_path = fold_dir / checkpoint_name
                if checkpoint_path.exists():
                    checkpoints.append(checkpoint_path)
                    break

        if checkpoints:
            return checkpoints

        # Pattern 2: Direct checkpoint in main folder
        for checkpoint_name in ["model_final_checkpoint.model", "model_best.model"]:
            checkpoint_path = self.model_folder / checkpoint_name
            if checkpoint_path.exists():
                checkpoints.append(checkpoint_path)

        if checkpoints:
            return checkpoints

        # Pattern 3: Checkpoint in "all" subfolder (common in Model Zoo)
        all_folder = self.model_folder / "all"
        if all_folder.exists():
            for checkpoint_name in ["model_final_checkpoint.model", "model_best.model"]:
                checkpoint_path = all_folder / checkpoint_name
                if checkpoint_path.exists():
                    checkpoints.append(checkpoint_path)

        if checkpoints:
            return checkpoints

        # Pattern 4: Search recursively for any .model file
        model_files = list(self.model_folder.glob("**/*.model"))
        if model_files:
            return model_files

        # Pattern 5: Search for model_*.pth files (some V1 variants)
        pth_files = list(self.model_folder.glob("**/model_*.pth"))
        if pth_files:
            return pth_files

        return checkpoints

    def _auto_detect_configuration(self) -> str:
        """Auto-detect the configuration from folder path or plans."""
        # FIRST: Check the folder path for explicit configuration
        folder_path_str = str(self.model_folder).lower().replace('\\', '/')

        print(f"Debug: Checking folder path: {folder_path_str}")

        # Check for 2d configuration
        if '/2d/' in folder_path_str or folder_path_str.endswith('/2d'):
            print("Debug: Detected 2d from folder path")
            return "2d"

        # Check for 3d_cascade_fullres
        if '3d_cascade_fullres' in folder_path_str:
            print("Debug: Detected 3d_cascade_fullres from folder path")
            return "3d_cascade_fullres"

        # Check for 3d_fullres
        if '3d_fullres' in folder_path_str:
            print("Debug: Detected 3d_fullres from folder path")
            return "3d_fullres"

        # Check for 3d_lowres
        if '3d_lowres' in folder_path_str:
            print("Debug: Detected 3d_lowres from folder path")
            return "3d_lowres"

        print("Debug: No configuration detected from folder path, checking plans...")

        # SECOND: If no folder hint, try to detect from plans
        if self.plans is None:
            print(
                "Warning: No plans found and no configuration in path, defaulting to 3d_fullres")
            return "3d_fullres"

        # V1 plans structure is different from V2
        # Try to detect from plans
        plans_per_stage = self.plans.get('plans_per_stage', {})

        print(f"Debug: plans_per_stage keys: {list(plans_per_stage.keys())}")

        # Check number of stages and their properties to determine configuration
        if len(plans_per_stage) == 2:
            # Two stages usually means cascade (lowres + fullres)
            print("Debug: 2 stages found, detecting as 3d_cascade_fullres")
            return "3d_cascade_fullres"
        elif len(plans_per_stage) == 1:
            # Single stage - check if it's 2D or 3D
            stage_data = list(plans_per_stage.values())[0]
            print(f"Debug: Stage data keys: {list(stage_data.keys())}")
            if 'num_pool_per_axis' in stage_data:
                num_pool = stage_data['num_pool_per_axis']
                print(
                    f"Debug: num_pool_per_axis: {num_pool}, length: {len(num_pool)}")
                # 2D models have 2 pooling axes, 3D models have 3
                if len(num_pool) == 2:
                    print("Debug: 2 pooling axes detected, returning 2d")
                    return "2d"
                else:
                    print("Debug: 3 pooling axes detected, returning 3d_fullres")
                    return "3d_fullres"

        # Default fallback
        print("Warning: Could not determine configuration from plans, defaulting to 3d_fullres")
        return "3d_fullres"

    def _auto_detect_folds(self) -> Tuple[int, ...]:
        """Auto-detect all available folds for ensemble prediction."""
        # First check if fold_all exists
        if (self.model_folder / "fold_all").exists():
            print("  Found fold_all directory")
            return "all"
        
        # Otherwise, detect individual fold directories
        fold_dirs = sorted(self.model_folder.glob("fold_*"))
        if fold_dirs:
            available_folds = []
            for fold_dir in fold_dirs:
                try:
                    fold_num = int(fold_dir.name.split("_")[1])
                    checkpoint = fold_dir / self.checkpoint_name
                    if not checkpoint.exists():
                        checkpoint = fold_dir / "checkpoint_best.pth"
                    if checkpoint.exists():
                        available_folds.append(fold_num)
                except (ValueError, IndexError):
                    continue
            
            if available_folds:
                available_folds.sort()
                print(f"  Found {len(available_folds)} folds: {available_folds}")
                return tuple(available_folds)
        
        print("  Warning: No folds detected, defaulting to fold_0")
        return (0,)

    def predict(self, image: np.ndarray, spacing: Tuple[float, ...],
                origin: Optional[Tuple[float, ...]] = None,
                direction: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run prediction using nnU-Net V1.

        For V1, we'll use a simplified approach similar to V2 but adapted for V1's API.

        Args:
            image: Input image array
            spacing: Voxel spacing
            origin: Image origin (optional)
            direction: Direction matrix (optional)

        Returns:
            Segmentation mask
        """
        if not self.models:
            raise ValueError("nnU-Net V1 model not loaded")

        print("\n=== nnU-Net V1 Prediction ===")
        print(f"Input shape: {image.shape}")
        print(f"Input spacing: {spacing}")
        print(f"Input dtype: {image.dtype}")
        print(f"Input range: [{np.min(image)}, {np.max(image)}]")

        try:
            # For V1, we'll use a similar file-based approach
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                # Create input and output directories
                input_dir = temp_dir / "input"
                output_dir = temp_dir / "output"
                input_dir.mkdir(exist_ok=True)
                output_dir.mkdir(exist_ok=True)

                # Save input as NIfTI
                input_file = input_dir / "input_0000.nii.gz"

                if origin is None:
                    origin = (0.0, 0.0, 0.0)
                if direction is None:
                    direction = np.eye(3)

                # Create SimpleITK image
                sitk_image = sitk.GetImageFromArray(image)
                sitk_image.SetSpacing(spacing[::-1])
                sitk_image.SetOrigin(
                    origin[::-1] if len(origin) == 3 else origin)
                sitk_image.SetDirection(direction.flatten().tolist())

                sitk.WriteImage(sitk_image, str(input_file))
                print(f"  Input saved to: {input_file.name}")

                # Run prediction using the loaded state dicts
                # For V1, we need to manually run the inference
                prediction = self._run_v1_inference(
                    str(input_file), str(output_dir))

                # The prediction should already be a numpy array
                if prediction is None:
                    raise ValueError("Prediction failed")

                unique_values = np.unique(prediction)
                print(f"\n✓ Inference complete")
                print(f"  Output shape: {prediction.shape}")
                print(f"  Output dtype: {prediction.dtype}")
                print(f"  Unique labels: {unique_values}")
                print(f"  Number of classes: {len(unique_values)}")
                print("=== Prediction Complete ===\n")

                return prediction.astype(np.uint8)

        except Exception as e:
            print(f"✗ Error during nnU-Net V1 prediction: {e}")
            raise

    def _run_v1_inference(self, input_file: str, output_dir: str) -> np.ndarray:
        """
        Run V1 inference using the loaded model.

        This is a simplified implementation. For full V1 support, you would need
        to import the actual nnU-Net V1 predictor classes.
        """
        print("\n⚠ nnU-Net V1 inference using compatibility mode")
        print("For best results, please use nnU-Net V2 or install nnunet package")

        # Try to use nnunet package if available
        try:

            print("  Using nnunet V1 package for prediction...")

            # V1 expects input folder with files
            input_folder = Path(input_file).parent

            # CRITICAL FIX: nnU-Net V1's predict_from_folder expects .model.pkl files
            # but Model Zoo downloads only have .model files
            # We need to check if .model.pkl files exist, and if not, skip using the package
            fold_dirs = list(self.model_folder.glob("fold_*"))
            has_pkl_files = False

            if fold_dirs:
                # Check first fold for .model.pkl file
                test_pkl = fold_dirs[0] / "model_final_checkpoint.model.pkl"
                has_pkl_files = test_pkl.exists()

            if not has_pkl_files:
                print("  ⚠ Model Zoo download detected (missing .model.pkl files)")
                print("  ⚠ The nnunet V1 package requires .model.pkl trainer files")
                print("  ⚠ Model Zoo downloads only include .model checkpoint files")
                print("\n  Falling back to manual inference...")
                return self._manual_v1_inference_modelzoo(input_file, output_dir)

            # If we have .model.pkl files, use the standard V1 inference
            # IMPORTANT: nnU-Net V1 appends ".model.pkl" to checkpoint_name
            # So if checkpoint is "model_final_checkpoint.model.pkl"
            # We should pass "model_final_checkpoint" (without .model or .model.pkl)
            checkpoint_base_name = self.checkpoint_name.replace(
                '.model.pkl', '').replace('.model', '')

            print(f"  Using checkpoint: {checkpoint_base_name}")

            predict_from_folder(
                str(self.model_folder),
                str(input_folder),
                str(output_dir),
                self.folds,
                save_npz=False,
                num_threads_preprocessing=1,
                num_threads_nifti_save=1,
                lowres_segmentations=None,
                part_id=0,
                num_parts=1,
                tta=False,
                mixed_precision=True,
                overwrite_existing=True,
                mode='normal',
                overwrite_all_in_gpu=None,
                step_size=0.5,
                checkpoint_name=checkpoint_base_name
            )

            # Load result
            output_file = Path(output_dir) / "input.nii.gz"
            if not output_file.exists():
                output_file = Path(output_dir) / "input_0000.nii.gz"

            if output_file.exists():
                result_img = sitk.ReadImage(str(output_file))
                segmentation = sitk.GetArrayFromImage(result_img)
                return segmentation
            else:
                raise FileNotFoundError(f"Output not found in {output_dir}")

        except ImportError:
            print("  ⚠ nnunet V1 package not installed")
            print("  To install: pip install nnunet")
            print("  Attempting manual inference (limited functionality)...")

            # Fallback: Load the image and try to do basic prediction
            return self._manual_v1_inference(input_file)
        except FileNotFoundError as e:
            if ".model.pkl" in str(e):
                print(f"  ⚠ Detected Model Zoo download (missing trainer pickle files)")
                print("  Falling back to manual inference...")
                return self._manual_v1_inference_modelzoo(input_file, output_dir)
            else:
                raise

    def _manual_v1_inference_modelzoo(self, input_file: str, output_dir: str) -> np.ndarray:
        """
        Manual inference for Model Zoo downloads that lack .model.pkl files.

        This implements a simplified V1 inference pipeline using only the .model checkpoints.
        """
        print("\n  === Manual V1 Inference (Model Zoo Mode) ===")

        try:
            # Import necessary V1 components
            print("  Loading image...")
            sitk_image = sitk.ReadImage(input_file)
            image_data = sitk.GetArrayFromImage(sitk_image)
            spacing = sitk_image.GetSpacing()[::-1]  # Reverse to (z, y, x)

            print(f"  Image shape: {image_data.shape}")
            print(f"  Spacing: {spacing}")

            # Load plans
            if self.plans is None:
                raise ValueError("Plans file required for manual inference")

            # Debug: print plans structure
            print(f"  Plans keys: {list(self.plans.keys())}")

            # Get stage parameters (use stage 1 = 3d_fullres or stage 0 = 3d_lowres)
            stage = 1 if self.configuration == "3d_fullres" else 0

            # Handle different plans formats
            if 'plans_per_stage' in self.plans:
                plans_stage = self.plans['plans_per_stage'][stage]
                print(
                    f"  Found plans_per_stage, stage keys: {list(plans_stage.keys())}")
            else:
                # Fallback: use plans directly
                plans_stage = self.plans
                print(
                    f"  Using plans directly, keys: {list(plans_stage.keys())}")

            print(
                f"  Using configuration: {self.configuration} (stage {stage})")
            print(f"  Patch size: {plans_stage.get('patch_size', 'N/A')}")

            # Extract network parameters with fallbacks
            net_num_pool_op_kernel_sizes = plans_stage.get('pool_op_kernel_sizes',
                                                           [[2, 2, 2]] * 5)
            net_conv_kernel_sizes = plans_stage.get('conv_kernel_sizes',
                                                    [[3, 3, 3]] * 6)

            # Get base_num_features with fallback
            base_num_features = plans_stage.get('base_num_features', 32)
            if base_num_features is None:
                base_num_features = 32  # Default for nnU-Net V1
                print(
                    f"  ⚠ base_num_features not found in plans, using default: {base_num_features}")

            # Get num_conv_per_stage with fallback
            num_conv_per_stage = plans_stage.get('num_conv_per_stage', 2)
            if num_conv_per_stage is None:
                num_conv_per_stage = 2
                print(
                    f"  ⚠ num_conv_per_stage not found in plans, using default: {num_conv_per_stage}")

            # Number of classes
            num_classes = self.plans.get(
                'num_classes', 1) + 1  # +1 for background
            num_modalities = self.plans.get('num_modalities', 1)

            print(f"  Number of classes (with background): {num_classes}")
            print(f"  Number of modalities: {num_modalities}")
            print(f"  Base features: {base_num_features}")
            print(f"  Conv per stage: {num_conv_per_stage}")
            print(f"  Pool sizes: {net_num_pool_op_kernel_sizes}")

            # Create network architecture
            print("  Creating network architecture...")
            device = self.models[0]['device']

            # Determine if 2D or 3D
            is_3d = '3d' in self.configuration

            # Use parameters from plans
            network = Generic_UNet(
                input_channels=num_modalities,
                base_num_features=base_num_features,
                num_classes=num_classes,
                num_pool=len(net_num_pool_op_kernel_sizes),
                num_conv_per_stage=num_conv_per_stage,
                feat_map_mul_on_downscale=2,
                conv_op=torch.nn.Conv3d if is_3d else torch.nn.Conv2d,
                norm_op=torch.nn.InstanceNorm3d if is_3d else torch.nn.InstanceNorm2d,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                dropout_op=torch.nn.Dropout3d if is_3d else torch.nn.Dropout2d,
                dropout_op_kwargs={'p': 0, 'inplace': True},
                nonlin=torch.nn.LeakyReLU,
                nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
                deep_supervision=True,
                dropout_in_localization=False,
                final_nonlin=lambda x: x,
                weightInitializer=None,
                pool_op_kernel_sizes=net_num_pool_op_kernel_sizes,
                conv_kernel_sizes=net_conv_kernel_sizes,
                upscale_logits=False,
                convolutional_pooling=True,
                convolutional_upsampling=True,
            )

            network.to(device)
            network.eval()

            print("  ✓ Network created")

            # Load weights from first checkpoint (fold 0)
            print(f"  Loading weights from checkpoint 0...")
            state_dict = self.models[0]['state_dict']

            # Try to load with strict=False first
            try:
                network.load_state_dict(state_dict, strict=True)
                print("  ✓ Weights loaded (strict)")
            except Exception as e:
                print(f"  ⚠ Strict loading failed: {e}")
                print("  Trying with strict=False...")
                network.load_state_dict(state_dict, strict=False)
                print("  ✓ Weights loaded (non-strict)")

            # Preprocess image
            print("  Preprocessing image...")

            # Normalize (z-score normalization)
            mean_val = image_data.mean()
            std_val = image_data.std()
            if std_val == 0:
                std_val = 1.0
            image_normalized = (image_data - mean_val) / std_val

            print(f"  Normalization: mean={mean_val:.2f}, std={std_val:.2f}")

            # Get target spacing if available
            target_spacing = plans_stage.get('current_spacing', spacing)
            print(f"  Target spacing: {target_spacing}")

            # Convert to torch tensor
            image_tensor = torch.from_numpy(image_normalized).float()
            image_tensor = image_tensor.unsqueeze(
                0).unsqueeze(0)  # Add batch and channel dims
            image_tensor = image_tensor.to(device)

            print(f"  Input tensor shape: {image_tensor.shape}")

            # Run inference
            print("  Running inference...")

            # Get patch size from plans
            patch_size = plans_stage.get('patch_size', [80, 192, 160])
            print(f"  Using sliding window with patch size: {patch_size}")

            # Use MONAI's sliding window inference if available, otherwise simple chunking
            try:

                print("  Using MONAI sliding window inference...")

                with torch.no_grad():
                    # MONAI's sliding window inference
                    output = monai_sliding_window(
                        inputs=image_tensor,
                        roi_size=patch_size,
                        sw_batch_size=1,
                        predictor=network,
                        overlap=0.5,
                        mode='gaussian',
                        device=device
                    )

                    # Handle deep supervision
                    if isinstance(output, (list, tuple)):
                        output = output[0]

                    # Apply softmax and argmax
                    output = F.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1)
                    prediction = prediction.cpu().numpy().squeeze()

            except ImportError:
                print("  MONAI not available, using simple patch-based inference...")
                prediction = self._simple_sliding_window(
                    network, image_tensor, patch_size, device
                )

            print(f"  ✓ Inference complete")
            print(f"  Output shape: {prediction.shape}")

            unique_vals = np.unique(prediction)
            print(f"  Unique values: {unique_vals}")
            print(
                f"  Value counts: {[(val, np.sum(prediction == val)) for val in unique_vals]}")

            # Save result
            output_file = Path(output_dir) / "input.nii.gz"
            result_sitk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            result_sitk.CopyInformation(sitk_image)
            sitk.WriteImage(result_sitk, str(output_file))

            print(f"  ✓ Result saved to {output_file.name}")

            return prediction.astype(np.uint8)

        except ImportError as e:
            print(f"  ✗ Failed to import nnunet V1 components: {e}")
            print(f"  Please install nnunet V1: pip install nnunet")
            raise NotImplementedError(
                "Manual V1 inference requires the 'nnunet' package. "
                "Install with: pip install nnunet"
            )
        except Exception as e:
            print(f"  ✗ Manual inference failed: {e}")
            raise

    def _manual_v1_inference(self, input_file: str) -> np.ndarray:
        """
        Attempt manual inference without nnunet package.
        This is a fallback and has limited functionality.
        """
        print("\n  ⚠ Using fallback manual inference")
        print("  This may not produce optimal results")

        # Load the input
        sitk_image = sitk.ReadImage(input_file)
        image_data = sitk.GetArrayFromImage(sitk_image)

        # This would require implementing the full V1 preprocessing and inference pipeline
        # For now, return a dummy prediction
        print("  ✗ Manual V1 inference not fully implemented")
        print("  Please install nnunet V1 package: pip install nnunet")

        raise NotImplementedError(
            "nnU-Net V1 requires the 'nnunet' package. "
            "Install with: pip install nnunet"
        )

    def get_num_classes(self) -> int:
        """Get number of output classes (including background)."""
        if self.dataset_json:
            # dataset_json labels includes background (0)
            return len(self.dataset_json.get('labels', {}))

        # Try to get from plans
        if self.plans:
            # V1 plans structure
            # num_classes in plans typically counts only foreground classes
            # so we add 1 for background
            num_classes = self.plans.get('num_classes', 0)
            if num_classes > 0:
                return num_classes + 1  # Add background class

        return 0

    def get_class_names(self) -> Dict[str, str]:
        """Get class names/labels."""
        if self.dataset_json:
            return self.dataset_json.get('labels', {})

        # Fallback: create generic labels
        num_classes = self.get_num_classes()
        if num_classes > 0:
            return {str(i): f"Class_{i}" for i in range(num_classes)}

        return {}

    def get_input_dimensions(self) -> str:
        """Get model input dimensions."""
        if self.configuration:
            return "3D" if "3d" in self.configuration else "2D"
        return "Unknown"

    def get_patch_size(self) -> Tuple[int, ...]:
        """Get the patch/tile size used for inference."""
        if self.plans and 'plans_per_stage' in self.plans:
            # Get stage based on configuration
            stage = 1 if self.configuration == "3d_fullres" else 0
            if stage in self.plans['plans_per_stage']:
                patch_size = self.plans['plans_per_stage'][stage].get(
                    'patch_size')
                if patch_size is not None:
                    return tuple(patch_size)
        return None

    def _simple_sliding_window(self, network, image_tensor, patch_size, device):
        """
        Simple sliding window inference without MONAI.
        Processes image in patches to avoid OOM errors.
        """

        # Get image dimensions
        _, _, d, h, w = image_tensor.shape
        pd, ph, pw = patch_size

        # Calculate number of patches needed
        step_d = pd // 2  # 50% overlap
        step_h = ph // 2
        step_w = pw // 2

        # Initialize output
        prediction = torch.zeros((d, h, w), dtype=torch.long, device='cpu')
        counts = torch.zeros((d, h, w), dtype=torch.float32, device='cpu')

        num_classes = 2  # Binary segmentation

        print(f"  Image shape: {d}×{h}×{w}")
        print(f"  Patch shape: {pd}×{ph}×{pw}")
        print(f"  Step size: {step_d}×{step_h}×{step_w}")

        # Iterate over patches
        total_patches = 0
        for z in range(0, d, step_d):
            for y in range(0, h, step_h):
                for x in range(0, w, step_w):
                    # Calculate patch boundaries
                    z_end = min(z + pd, d)
                    y_end = min(y + ph, h)
                    x_end = min(x + pw, w)

                    z_start = max(0, z_end - pd)
                    y_start = max(0, y_end - ph)
                    x_start = max(0, x_end - pw)

                    # Extract patch
                    patch = image_tensor[:, :, z_start:z_end,
                                         y_start:y_end, x_start:x_end]

                    # Run inference on patch
                    with torch.no_grad():
                        patch_output = network(patch)

                        # Handle deep supervision
                        if isinstance(patch_output, (list, tuple)):
                            patch_output = patch_output[0]

                        # Softmax
                        patch_output = F.softmax(patch_output, dim=1)

                        # Argmax
                        patch_pred = torch.argmax(
                            patch_output, dim=1).squeeze().cpu()

                    # Add to output (weighted by counts for overlap)
                    prediction[z_start:z_end, y_start:y_end,
                               x_start:x_end] += patch_pred
                    counts[z_start:z_end, y_start:y_end, x_start:x_end] += 1

                    total_patches += 1
                    if total_patches % 10 == 0:
                        print(f"    Processed {total_patches} patches...")

        print(f"  Total patches processed: {total_patches}")

        # Average predictions where patches overlapped
        prediction = (prediction / counts).round().long().numpy()

        return prediction

    def switch_device(self, device: torch.device):
        """
        Switch the device (CPU/GPU) for the predictor.
        
        Args:
            device: New torch device (cpu or cuda)
        """
        # SHow a warning that switching device is not supported in V1
        print("\n⚠ Warning: Switching device is not fully supported in nnU-Net V1")
        print("Please reload the model on the desired device.\n")