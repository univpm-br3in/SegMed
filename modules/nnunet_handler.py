"""
nnU-Net V2 Handler for SegMed - Using Official Predictor
Handles loading and inference with nnU-Net V2 models using nnUNetPredictor
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import SimpleITK as sitk
from glob import glob

# Suppress nnU-Net environment warnings (not needed for inference-only usage)
# These paths are only required for training/preprocessing, not for running predictions
if 'nnUNet_raw' not in os.environ:
    os.environ['nnUNet_raw'] = os.path.join(tempfile.gettempdir(), 'nnunet_raw')
if 'nnUNet_preprocessed' not in os.environ:
    os.environ['nnUNet_preprocessed'] = os.path.join(tempfile.gettempdir(), 'nnunet_preprocessed')
if 'nnUNet_results' not in os.environ:
    os.environ['nnUNet_results'] = os.path.join(tempfile.gettempdir(), 'nnunet_results')


class NnUNetV2Handler:
    """Handler for nnU-Net V2 models using the official nnUNetPredictor."""
    
    def __init__(self):
        self.predictor = None
        self.model_folder = None
        self.dataset_json = None
        self.plans = None
        self.configuration = None  # '2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'
        self.folds = None
        self.checkpoint_name = 'checkpoint_final.pth'
        
    def is_nnunet_folder(self, folder_path: str) -> bool:
        """
        Check if the folder is a valid nnU-Net V2 trained model folder.
        
        Required structure (minimum):
        folder/
        ├── plans.json (or nnUNetPlans.json)  ← REQUIRED
        └── fold_X/
            └── checkpoint_final.pth (or checkpoint_best.pth)  ← REQUIRED
        
        Optional:
        ├── dataset.json  ← OPTIONAL (provides class labels/metadata)
        """
        folder = Path(folder_path)
        
        # Check for plans file (REQUIRED)
        plans_json = folder / "plans.json"
        alt_plans_json = folder / "nnUNetPlans.json"
        if not plans_json.exists() and not alt_plans_json.exists():
            print (f"✗ Missing plans.json or nnUNetPlans.json in {folder_path}")
            return False
        
        # Check for at least one fold directory with checkpoint (REQUIRED)
        fold_dirs = list(folder.glob("fold_*"))
        if not fold_dirs:
            print(f"✗ No fold directories found in {folder_path}")
            return False
        
        # Check if at least one fold has a checkpoint
        for fold_dir in fold_dirs:
            if glob(str(fold_dir / "*.pth")):
                # If any .pth file exists, assume it's a checkpoint
                return True  
            print(f"✗ No checkpoint found in {fold_dir}")
        
        return False
    
    def load_nnunet_model(self, folder_path: str, device: torch.device, 
                         folds: Optional[Tuple[int, ...]] = None) -> bool:
        """
        Load nnU-Net V2 model using the official nnUNetPredictor.
        
        Args:
            folder_path: Path to nnU-Net trained model folder
            device: torch device (cpu or cuda)
            folds: Tuple of folds to use (e.g., (0,) or (0, 1, 2, 3, 4) for ensemble)
                  If None, will use fold_0 by default
            checkpoint_name: Name of checkpoint to load ('checkpoint_final.pth' or 'checkpoint_best.pth')
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            
            print("\n=== Loading nnU-Net V2 Model ===")
            print(f"Model path: {folder_path}")
            
            self.model_folder = Path(folder_path)
            
            pth_files = glob(str(self.model_folder / "fold_*" / "*.pth"))
            if "best" in [Path(f).name for f in pth_files]:
                # find THE EXACT name of the best checkpoint
                for f in pth_files:
                    if "best" in Path(f).name:
                        self.checkpoint_name = Path(f).name
                        break
            else:
                self.checkpoint_name = os.path.basename(pth_files[0])
            
            # Load dataset.json for information (OPTIONAL)
            dataset_json_path = self.model_folder / "dataset.json"
            if dataset_json_path.exists():
                with open(dataset_json_path, 'r') as f:
                    self.dataset_json = json.load(f)
                print(f"✓ Dataset JSON loaded")
                print(f"  Dataset: {self.dataset_json.get('name', 'Unknown')}")
                print(f"  Modalities: {self.dataset_json.get('modality', {})}")
                print(f"  Labels: {self.dataset_json.get('labels', {})}")
            else:
                print(f"⚠ Dataset JSON not found (optional) - class labels will be unavailable")
                self.dataset_json = None
            
            # Load plans.json for information (REQUIRED)
            plans_path = self.model_folder / "plans.json"
            if not plans_path.exists():
                plans_path = self.model_folder / "nnUNetPlans.json"
            
            with open(plans_path, 'r') as f:
                self.plans = json.load(f)
            print(f"✓ Plans loaded")
            
            # Auto-detect configuration
            self.configuration = self._auto_detect_configuration()
            print(f"  Configuration: {self.configuration}")
            
            # Auto-detect folds if not specified
            if folds is None:
                folds = self._auto_detect_folds()
            self.folds = folds
            print(f"  Using folds: {self.folds}")
            
            # Set device string for nnUNetPredictor
            device_str = 'cuda' if device.type == 'cuda' else 'cpu'
            
            # Initialize nnUNetPredictor
            print(f"  Initializing nnUNetPredictor on {device_str}...")
            self.predictor = nnUNetPredictor(
                tile_step_size=0.5,  # Overlap for sliding window
                use_gaussian=True,   # Gaussian importance weighting
                use_mirroring=True,  # Test-time augmentation with mirroring
                perform_everything_on_device=True,  # Keep everything on GPU if available
                device=torch.device(device_str),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
            # Initialize the predictor with model folder and folds
            self.predictor.initialize_from_trained_model_folder(
                str(self.model_folder),
                use_folds=self.folds,
                checkpoint_name=self.checkpoint_name
            )
            
            print(f"✓ nnU-Net V2 model loaded successfully")
            print(f"  Architecture: {self.predictor.network.__class__.__name__}")
            print(f"  Number of classes: {self.get_num_classes()}")
            print(f"  Tile size: {self.predictor.configuration_manager.patch_size}")
            print("=== nnU-Net V2 Model Ready ===\n")
            
            return True
            
        except ImportError as e:
            print(f"✗ Error: nnunetv2 package not installed")
            print(f"  Please install with: pip install nnunetv2")
            print(f"  {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading nnU-Net model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _auto_detect_configuration(self) -> str:
        """Auto-detect the best configuration from plans."""
        configs = self.plans.get('configurations', {})
        
        # Priority order: 3d_fullres > 3d_lowres > 2d
        if '3d_fullres' in configs:
            return '3d_fullres'
        elif '3d_lowres' in configs:
            return '3d_lowres'
        elif '2d' in configs:
            return '2d'
        else:
            # Return first available
            return list(configs.keys())[0]
    
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
                direction: Optional[np.ndarray] = None,
                properties: Optional[Dict] = None) -> np.ndarray:
        """
        Run full prediction pipeline using nnUNetPredictor.
        
        The nnUNetPredictor handles ALL preprocessing and postprocessing automatically:
        - Normalization (based on plans)
        - Resampling to target spacing
        - Transposition
        - Sliding window inference
        - Argmax
        - Resampling back to original spacing
        
        Args:
            image: Input image array in (x, y, z) order
            spacing: Voxel spacing (sx, sy, sz)
            origin: Image origin in physical (x, y, z) coordinates
            direction: Direction cosine matrix (3x3 or flattened 9-element array)
                       in ITK/LPS coordinates
            properties: Optional properties dict (if None, will be auto-generated)
        
        Returns:
            Segmentation mask in original image space
        """
        if self.predictor is None:
            raise ValueError("nnU-Net model not loaded")
        
        print("\n=== nnU-Net V2 Prediction ===")
        print(f"Input shape: {image.shape}")
        print(f"Input spacing: {spacing}")
        print(f"Input dtype: {image.dtype}")
        print(f"Input range: [{np.min(image)}, {np.max(image)}]")
        
        try:
            import SimpleITK as sitk
            import tempfile
            from pathlib import Path
            
            # Create temporary directory for nnU-Net processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Create separate input and output directories to avoid confusion
                input_dir = temp_dir / "input"
                output_dir = temp_dir / "output"
                input_dir.mkdir(exist_ok=True)
                output_dir.mkdir(exist_ok=True)
                
                # Save image as NIfTI (nnU-Net's preferred format)
                input_file = input_dir / "temp_input_0000.nii.gz"
                
                # Set default values if not provided
                if origin is None:
                    origin = (0.0, 0.0, 0.0)
                if direction is None:
                    direction = np.eye(3)

                if np.asarray(image).ndim != 3:
                    raise ValueError(f"Expected a 3D image, got {np.asarray(image).shape}")

                spacing = tuple(float(v) for v in spacing)
                origin = tuple(float(v) for v in origin)
                direction = np.asarray(direction, dtype=float).reshape(3, 3)
                direction_itk = direction.ravel().tolist()

                # SegMed uses (x, y, z); SimpleITK reads NumPy arrays as (z, y, x).
                image_zyx = np.ascontiguousarray(
                    np.asarray(image, dtype=np.float32).transpose(2, 1, 0)
                )
                sitk_image = sitk.GetImageFromArray(image_zyx)
                sitk_image.SetSpacing(spacing)
                sitk_image.SetOrigin(origin)
                sitk_image.SetDirection(direction_itk)
                
                print(f"  Image metadata:")
                print(f"    Spacing: {spacing}")
                print(f"    Origin: {origin}")
                print(f"    Direction: {direction.flatten() if isinstance(direction, np.ndarray) else direction}")
                
                # Save to file
                sitk.WriteImage(sitk_image, str(input_file))
                print(f"  Temporary input created: {input_file.name}")
                
                # Run prediction using nnUNetPredictor.predict_from_files
                # Note: num_processes_segmentation_export must be >= 1
                print("  Running inference with automatic preprocessing/postprocessing...")
                
                self.predictor.predict_from_files(
                    [[str(input_file)]],  # List of cases, each with list of modality files
                    [str(output_dir)],    # Output filename prefix
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=1,
                    num_processes_segmentation_export=1,  # Must be >= 1
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0
                )
                
                print("  Inference complete, checking for output files...")
                
                # Give the background worker time to finish writing
                import time
                time.sleep(1.0)  # Increased wait time
                
                # Load the result
                # The list-based output argument is a filename prefix, not a folder.
                # The requested prefix here produces temp_dir/output.nii.gz.
                
                # First, check both possible locations
                print(f"  Searching for output file...")
                print(f"  temp_dir: {temp_dir}")
                print(f"  output_dir: {output_dir}")
                
                # List all files in temp directory (parent of output_dir) for debugging
                all_temp_files = list(temp_dir.glob('**/*'))  # Recursive search
                print(f"  All files in temp directory tree: {[str(f.relative_to(temp_dir)) for f in all_temp_files if f.is_file()]}")
                
                # List files in output directory
                all_output_files = list(output_dir.glob('**/*'))  # Recursive search
                nifti_files = [f for f in all_output_files if f.suffix in ['.gz', '.nii']]
                print(f"  Files in output directory: {[f.name for f in all_output_files if f.is_file()]}")
                print(f"  NIfTI files in output dir: {[f.name for f in nifti_files]}")
                
                # nnU-Net creates the output with the folder name in the parent directory
                # output_dir.name = "output", so it creates temp_dir/output.nii.gz
                output_file = temp_dir / f"{output_dir.name}.nii.gz"
                print(f"  Expected output file (folder-based naming): {output_file}")
                print(f"  Expected output exists: {output_file.exists()}")
                
                if not output_file.exists():
                    # Try alternative naming patterns
                    print("  Primary output file not found, trying alternatives...")
                    alternatives = [
                        output_dir / "temp_input.nii.gz",                # Inside output dir (old assumption)
                        temp_dir / "temp_input.nii.gz",                  # In parent temp dir
                        temp_dir / f"{temp_dir.name}.nii.gz",           # Using temp folder name
                        output_dir / f"{output_dir.name}.nii.gz",       # Inside output, using its name
                    ]
                    
                    for alt in alternatives:
                        print(f"    Checking: {alt.relative_to(temp_dir) if alt.is_relative_to(temp_dir) else alt}")
                        if alt.exists():
                            output_file = alt
                            print(f"  ✓ Found output at: {alt.relative_to(temp_dir)}")
                            break
                    
                    # If still not found, search all NIfTI files in temp directory tree
                    if not output_file.exists():
                        print("  Still not found, searching all NIfTI files in temp tree...")
                        nifti_files = list(temp_dir.glob('**/*.nii.gz')) + list(temp_dir.glob('**/*.nii'))
                        
                        if nifti_files:
                            print(f"  Found {len(nifti_files)} NIfTI files total:")
                            
                            # Try to find the segmentation by checking content
                            for nii_file in nifti_files:
                                # Skip the input file
                                if 'temp_input_0000' in nii_file.name:
                                    print(f"    Skipping input: {nii_file.relative_to(temp_dir)}")
                                    continue
                                
                                try:
                                    test_img = sitk.ReadImage(str(nii_file))
                                    test_data = sitk.GetArrayFromImage(test_img)
                                    test_unique = np.unique(test_data)
                                    
                                    print(f"    Checking {nii_file.relative_to(temp_dir)}: {len(test_unique)} unique values, range [{test_data.min():.1f}-{test_data.max():.1f}]")
                                    
                                    # Segmentation should have few unique values
                                    if len(test_unique) <= 20 and test_data.max() < 100:
                                        print(f"  ✓✓ This looks like a segmentation!")
                                        output_file = nii_file
                                        break
                                except Exception as e:
                                    print(f"    Error reading: {e}")
                                    continue
                
                if not output_file.exists():
                    raise FileNotFoundError(
                        f"nnU-Net output file not found.\n"
                        f"Expected: {temp_dir.name}/output.nii.gz\n"
                        f"Also tried: temp_input.nii.gz in various locations\n"
                        f"All files in temp tree: {[str(f.relative_to(temp_dir)) for f in all_temp_files if f.is_file()]}"
                    )
                
                print(f"  Reading segmentation from: {output_file.name}")
                
                # Read segmentation
                result_img = sitk.ReadImage(str(output_file))
                segmentation = sitk.GetArrayFromImage(result_img)
                
                # Verify we got a segmentation, not the input image
                unique_values = np.unique(segmentation)
                print(f"  Raw output dtype: {segmentation.dtype}")
                print(f"  Raw output range: [{segmentation.min()}, {segmentation.max()}]")
                print(f"  Number of unique values: {len(unique_values)}")
                
                # Check if this looks like the input image (bad!)
                if len(unique_values) > 20 and segmentation.max() > 100:
                    print(f"\n  ⚠⚠⚠ WARNING: Output has {len(unique_values)} unique values")
                    print(f"  ⚠ This looks like the INPUT image, not segmentation!")
                    print(f"  ⚠ Range: [{segmentation.min()}, {segmentation.max()}]")
                    print(f"  ⚠ Reading wrong file: {output_file.relative_to(temp_dir)}")
                    
                    # Try to find the actual segmentation file
                    print(f"\n  Searching entire temp tree for actual segmentation file...")
                    
                    # Check all NIfTI files in temp directory tree
                    found_segmentation = False
                    for nii_file in temp_dir.glob('**/*.nii*'):
                        if nii_file == output_file:
                            continue  # Skip the one we already tried
                        
                        # Skip the input file
                        if 'temp_input_0000' in nii_file.name:
                            print(f"  Skipping input file: {nii_file.relative_to(temp_dir)}")
                            continue
                        
                        try:
                            test_img = sitk.ReadImage(str(nii_file))
                            test_data = sitk.GetArrayFromImage(test_img)
                            test_unique = np.unique(test_data)
                            
                            print(f"  Checking {nii_file.relative_to(temp_dir)}:")
                            print(f"    Unique values: {len(test_unique)}, Range: [{test_data.min()}, {test_data.max()}]")
                            
                            # If this looks like a segmentation (few unique values, small range)
                            if len(test_unique) <= 20 and test_data.max() < 100:
                                print(f"  ✓✓✓ This looks like a segmentation! Using it.")
                                segmentation = test_data
                                output_file = nii_file
                                unique_values = test_unique
                                found_segmentation = True
                                break
                        except Exception as e:
                            print(f"    Error reading {nii_file.name}: {e}")
                            continue
                    
                    if not found_segmentation:
                        print(f"\n  ✗ Could not find valid segmentation file!")
                        print(f"  ✗ All files in temp tree: {[str(f.relative_to(temp_dir)) for f in all_temp_files if f.is_file()]}")
                
                # Convert the SimpleITK array back to SegMed's (x, y, z) order.
                expected_shape = tuple(np.asarray(image).shape[::-1])
                if segmentation.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected output shape {segmentation.shape}; "
                        f"expected {expected_shape} in SimpleITK order"
                    )
                segmentation = np.ascontiguousarray(segmentation.transpose(2, 1, 0))

                print(f"\n✓ Inference complete")
                print(f"  Final output file: {output_file.relative_to(temp_dir) if output_file.is_relative_to(temp_dir) else output_file.name}")
                print(f"  Output shape: {segmentation.shape}")
                print(f"  Output dtype: {segmentation.dtype}")
                print(f"  Unique labels: {unique_values}")
                print(f"  Number of classes: {len(unique_values)}")
                print("=== Prediction Complete ===\n")
                
                # Convert to integer labels
                return segmentation.astype(np.uint8)
                
        except ImportError as e:
            print(f"✗ Error: SimpleITK not installed")
            print(f"  Please install with: pip install SimpleITK")
            raise
        
        except Exception as e:
            print(f"✗ Error during nnU-Net prediction: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_from_nifti(self, nifti_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Predict directly from NIfTI file using nnUNetPredictor.
        
        This is the recommended way to use nnU-Net as it preserves all metadata.
        
        Args:
            nifti_path: Path to input NIfTI file
            output_path: Optional path to save output (if None, just return array)
        
        Returns:
            Segmentation mask
        """
        if self.predictor is None:
            raise ValueError("nnU-Net model not loaded")
        
        print("\n=== nnU-Net V2 Prediction from NIfTI ===")
        print(f"Input: {nifti_path}")
        
        # Create temporary output folder if needed
        if output_path is None:
            temp_dir = tempfile.mkdtemp()
            output_folder = temp_dir
        else:
            output_folder = str(Path(output_path).parent)
        
        # Run prediction
        self.predictor.predict_from_files(
            [[nifti_path]],  # List of cases, each with list of modality files
            [output_folder],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
        
        # Load the result
        output_file = Path(output_folder) / f"{Path(nifti_path).stem}.nii.gz"
        if not output_file.exists():
            output_file = Path(output_folder) / f"{Path(nifti_path).stem}_0000.nii.gz"
        
        if output_file.exists():
            result_img = sitk.ReadImage(str(output_file))
            segmentation = sitk.GetArrayFromImage(result_img)
            print(f"✓ Prediction saved to: {output_file}")
            print("=== Prediction Complete ===\n")
            return segmentation
        else:
            raise FileNotFoundError(f"Output file not found: {output_file}")
    
    def get_input_dimensions(self) -> str:
        """Get model input dimensions."""
        if self.configuration:
            return "3D" if "3d" in self.configuration else "2D"
        return "Unknown"
    
    def get_num_classes(self) -> int:
        """
        Get number of output classes.
        
        Tries to get from dataset.json first, falls back to predictor's label_manager.
        """
        if self.dataset_json:
            return len(self.dataset_json.get('labels', {}))
        
        # Fallback: try to get from predictor's label_manager
        if self.predictor and hasattr(self.predictor, 'label_manager'):
            if hasattr(self.predictor.label_manager, 'all_labels'):
                return len(self.predictor.label_manager.all_labels)
        
        # Last resort: try to get from plans
        if self.plans and 'num_classes' in self.plans:
            return self.plans['num_classes']
        
        return 0
    
    def get_class_names(self) -> Dict[str, str]:
        """
        Get class names/labels.
        
        Returns labels from dataset.json if available, otherwise returns generic labels.
        """
        if self.dataset_json:
            return self.dataset_json.get('labels', {})
        
        # Fallback: create generic labels based on number of classes
        num_classes = self.get_num_classes()
        if num_classes > 0:
            return {str(i): f"Class_{i}" for i in range(num_classes)}
        
        return {}
    
    def get_patch_size(self) -> Tuple[int, ...]:
        """Get the patch/tile size used for inference."""
        if self.predictor and hasattr(self.predictor, 'configuration_manager'):
            return tuple(self.predictor.configuration_manager.patch_size)
        return None
    
    def enable_ensemble(self, use_all_folds: bool = True):
        """
        Enable ensemble prediction using multiple folds.
        
        Args:
            use_all_folds: If True, use all available folds. If False, use only fold 0.
        
        Note: This requires reloading the model.
        """
        if use_all_folds:
            available_folds = sorted([int(f.name.split('_')[1]) 
                                     for f in self.model_folder.glob("fold_*")])
            self.folds = tuple(available_folds)
            print(f"Ensemble mode: Using folds {self.folds}")
        else:
            self.folds = (0,)
            print("Single fold mode: Using fold 0")
    
    def switch_device(self, device: torch.device):
        """
        Switch the device (CPU/GPU) for the predictor.
        
        Args:
            device: New torch device (cpu or cuda)
        
        Note: This moves the model to the new device without reloading from disk.
        """
        if self.predictor is None:
            raise ValueError("Model not loaded yet")
        
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        print(f"\n=== Switching nnU-Net to {device_str.upper()} ===")
        
        # Update predictor device
        self.predictor.device = torch.device(device_str)
        
        # Move network to new device
        if hasattr(self.predictor, 'network'):
            self.predictor.network.to(device)
            print(f"✓ Model moved to {device_str}")
        
        # Move list of parameters if they exist
        if hasattr(self.predictor, 'list_of_parameters'):
            for params in self.predictor.list_of_parameters:
                for param_list in params:
                    for p in param_list:
                        if isinstance(p, torch.nn.Module):
                            p.to(device)
        
        print(f"=== Device switch complete ===\n")