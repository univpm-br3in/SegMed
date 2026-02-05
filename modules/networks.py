"""
Clean Network Module for SegMed
Supports Keras, PyTorch, and MONAI Bundle models
"""

import numpy as np
import torch
import tensorflow as tf
import os
import json
import nibabel as nib
import tempfile
from scipy.ndimage import zoom
from PySide6.QtWidgets import QFileDialog, QMessageBox
from torchinfo import summary
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from modules.nnunet_handler import NnUNetV2Handler
from modules.nnunet_v1_handler import NnUNetV1Handler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Network:
    def __init__(self):
        self.input_shape = None
        self.model = None
        self.processing_type = "cpu"
        self.input_channels = None
        self.input_dims = None  # 2D or 3D
        self.model_type = None  # 'keras', 'pytorch', 'monai', or 'nnunet'
        self.monai_bundle = None
        self.monai_preprocessing = None
        self.monai_postprocessing = None
        self.nnunet_handler = None  # nnU-Net V2 handler
        self.nnunet_v1_handler = None  # nnU-Net V1 handler
        self.nnunet_version = None  # Track which version (1 or 2)

    def set_processing_type(self, processing_type):
        """Set processing device (CPU or GPU)."""
        if processing_type.lower() == "cpu":
            self.processing_type = "cpu"
        elif processing_type.lower() == "gpu":
            if torch.cuda.is_available():
                self.processing_type = "gpu"
            else:
                raise ValueError("GPU is not available. Please use CPU.")

    def load_model(self, model_path, weights_path=None):
        """Load a model (Keras, PyTorch, or MONAI Bundle).

        Args:
            model_path: Path to the model file or directory
            weights_path: Optional path to weights file (for TorchScript models)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        self.model_type = None

        # Check for nnU-Net or MONAI bundle
        if os.path.isdir(model_path):
            if self._is_nnunet_folder(model_path):
                return self._load_nnunet_model(model_path)
            elif self._is_monai_bundle(model_path):
                return self._load_monai_bundle(model_path)

        # Keras models
        if model_path.endswith('.json'):
            return self._load_keras_json(model_path)
        elif model_path.endswith('.yaml') or model_path.endswith('.yml'):
            return self._load_keras_yaml(model_path)
        elif model_path.endswith('.h5') or model_path.endswith('.keras'):
            return self._load_keras_h5(model_path)

        # PyTorch models
        elif model_path.endswith('.ts'):
            return self._load_pytorch_ts(model_path, weights_path)
        elif model_path.endswith(('.pt', '.pth', '.pth.tar')):
            return self._load_pytorch_checkpoint(model_path)

        raise ValueError(f"Unsupported model format: {model_path}")

    def _is_monai_bundle(self, path):
        """Check if directory is a MONAI bundle."""
        if not os.path.isdir(path):
            return False
        configs_dir = os.path.join(path, "configs")
        models_dir = os.path.join(path, "models")
        has_configs = os.path.isdir(configs_dir)
        has_models = os.path.isdir(models_dir)
        has_inference_config = os.path.exists(
            os.path.join(configs_dir, "inference.json"))
        return has_configs and (has_models or has_inference_config)

    def _load_monai_bundle(self, bundle_path):
        """Load any MONAI bundle with unified approach."""
        try:
            from monai.bundle import ConfigParser
            import sys

            print(f"Loading MONAI bundle from: {bundle_path}")

            # Find config file
            config_file = os.path.join(
                bundle_path, "configs", "inference.json")
            if not os.path.exists(config_file):
                config_file = os.path.join(
                    bundle_path, "configs", "inference.yaml")
            if not os.path.exists(config_file):
                raise ValueError(
                    f"No inference configuration found in {bundle_path}/configs/")

            device = self._get_torch_device()
            print(f"Using device: {device}")

            # Add bundle path to Python path for custom modules
            if bundle_path not in sys.path:
                sys.path.insert(0, bundle_path)
                print(f"Added bundle path to sys.path")

            # Change to bundle directory
            original_dir = os.getcwd()
            os.chdir(bundle_path)

            try:
                # Parse config
                parser = ConfigParser()
                parser.read_config(config_file)

                # Fix MONAI parameter compatibility issues (img_size -> spatial_size, etc.)
                self._fix_monai_compatibility(parser)

                # Override ALL device references
                print("\nOverriding device settings in config...")
                self._override_all_devices_in_config(parser, str(device))
                print(f"✓ All device references set to: {device}")

                # Detect model type for informational purposes
                model_type = self._detect_monai_model_architecture(parser)
                print(f"\nDetected model type: {model_type}")

                # STEP 1: Load architecture from config (WITHOUT weights)
                print("\nLoading model architecture from config...")

                # Patch torch.load to prevent weight loading during architecture creation
                original_torch_load = torch.load
                weight_files_loaded = []

                def patched_torch_load(*args, **kwargs):
                    # Track what files are being loaded
                    if args:
                        weight_files_loaded.append(str(args[0]))
                    kwargs['weights_only'] = False
                    kwargs['map_location'] = device
                    return original_torch_load(*args, **kwargs)

                torch.load = patched_torch_load

                try:
                    if 'network_def' in parser:
                        self.model = parser.get_parsed_content('network_def')
                    elif 'network' in parser:
                        self.model = parser.get_parsed_content('network')
                    else:
                        raise ValueError(
                            "No network_def or network found in config")

                    print(f"✓ Model architecture loaded")
                    print(f"  Type: {type(self.model).__name__}")
                    print(
                        f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

                finally:
                    torch.load = original_torch_load

                # STEP 2: Manually load weights (ALWAYS, regardless of model type)
                print("\nManually loading weights...")
                weights_loaded = self._manually_load_weights(
                    bundle_path, device, weight_files_loaded)

                if not weights_loaded:
                    print("⚠ Warning: Could not manually load weights")
                    print("  Using weights from ConfigParser (may be incorrect)")

                # Move to device and set to eval mode
                self.model.to(device)
                self.model.eval()

                # Load preprocessing and postprocessing
                print("\nLoading transforms...")
                try:
                    if 'preprocessing' in parser:
                        self.monai_preprocessing = parser.get_parsed_content(
                            'preprocessing')
                        print("✓ Preprocessing loaded")

                    if 'postprocessing' in parser:
                        self.monai_postprocessing = parser.get_parsed_content(
                            'postprocessing')

                        # Ensure correct postprocessing order
                        self._ensure_softmax_in_postprocessing()

                        print("✓ Postprocessing loaded")
                except Exception as e:
                    print(f"⚠ Warning: Could not load transforms: {e}")
                    self.monai_preprocessing = None
                    self.monai_postprocessing = None

            finally:
                os.chdir(original_dir)
                # Remove bundle path from sys.path to avoid conflicts
                if bundle_path in sys.path:
                    sys.path.remove(bundle_path)

            # Store bundle info
            self.monai_bundle = {'path': bundle_path, 'config': parser}
            self.model_type = 'monai'
            self.input_dims = "3D"

            print(f"\n✓ Successfully loaded MONAI bundle")
            print(f"  Model: {type(self.model).__name__}")
            print(
                f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to load MONAI bundle: {e}")

    def _is_nnunet_folder(self, path):
        """
        Check if directory is an nnU-Net trained model folder.
        Returns tuple: (is_nnunet, version) where version is 1, 2, or None
        """
        if not os.path.isdir(path):
            return False, None

        # Check for V2 first (more common now)
        v2_handler = NnUNetV2Handler()
        if v2_handler.is_nnunet_folder(path):
            return True, 2

        # Check for V1
        v1_handler = NnUNetV1Handler()
        if v1_handler.is_nnunet_v1_folder(path):
            return True, 1

        return False, None

    def _load_nnunet_model(self, folder_path):
        """
        Load nnU-Net model (auto-detects V1 or V2).
        """
        try:
            # Auto-detect version
            is_nnunet, version = self._is_nnunet_folder(folder_path)

            if not is_nnunet:
                print("✗ Not a valid nnU-Net folder")
                return False

            if version == 2:
                return self._load_nnunet_v2(folder_path)
            elif version == 1:
                return self._load_nnunet_v1(folder_path)
            else:
                print("✗ Unknown nnU-Net version")
                return False

        except Exception as e:
            print(f"✗ Error loading nnU-Net model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_nnunet_v2(self, folder_path):
        """Load nnU-Net V2 model using official nnUNetPredictor."""
        try:
            print(f"\n=== Loading nnU-Net V2 Model ===")

            # Create handler
            self.nnunet_handler = NnUNetV2Handler()
            self.nnunet_version = 2

            # Get device
            device = self._get_torch_device()

            success = self.nnunet_handler.load_nnunet_model(
                folder_path,
                device,
                folds=None,  # Auto-detect all folds for ensemble
                checkpoint_name='checkpoint_final.pth'
            )

            if not success:
                return False

            # Set model type and dimensions
            self.model_type = 'nnunet'
            self.model = self.nnunet_handler.predictor.network
            self.input_dims = self.nnunet_handler.get_input_dimensions()

            print(f"✓ nnU-Net V2 model loaded successfully")

            return True

        except Exception as e:
            print(f"✗ Error loading nnU-Net V2 model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_nnunet_v1(self, folder_path):
        """Load nnU-Net V1 model."""
        try:
            print(f"\n=== Loading nnU-Net V1 Model ===")

            # Create handler
            self.nnunet_v1_handler = NnUNetV1Handler()
            self.nnunet_version = 1

            # Get device
            device = self._get_torch_device()

            success = self.nnunet_v1_handler.load_nnunet_v1_model(
                folder_path,
                device,
                folds=None,  # Auto-detect all folds for ensemble
                checkpoint_name='model_final_checkpoint.model'
            )

            if not success:
                return False

            # Set model type and dimensions
            self.model = 1  # just a placeholder to avoid errors
            self.model_type = 'nnunet'
            self.input_dims = self.nnunet_v1_handler.get_input_dimensions()

            print(f"✓ nnU-Net V1 model loaded successfully")

            return True

        except Exception as e:
            print(f"✗ Error loading nnU-Net V1 model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _fix_monai_compatibility(self, parser):
        """
        Fix SwinUNETR parameters for MONAI 1.5+:
        - Remove deprecated 'img_size' parameter (removed in MONAI 1.5)
        - The model now handles dynamic input sizes during forward()
        """
        try:
            # Get the network definition config
            if 'network_def' in parser:
                network_config = parser['network_def']
            elif 'network' in parser:
                network_config = parser['network']
            else:
                return  # No network config to fix

            # Check if this is a SwinUNETR model
            if isinstance(network_config, dict):
                model_class = network_config.get('_target_', '')

                # If it's a SwinUNETR, remove the img_size parameter
                if 'SwinUNETR' in model_class:
                    if 'img_size' in network_config:
                        print(
                            f"  ⚠ Removing deprecated 'img_size' parameter from SwinUNETR config")
                        print(
                            f"    (img_size was removed in MONAI 1.5 - model now accepts dynamic sizes)")
                        del network_config['img_size']

                    # Also check for spatial_shape which might be used in some configs
                    if 'spatial_shape' in network_config:
                        print(f"  ⚠ Removing 'spatial_shape' parameter")
                        del network_config['spatial_shape']

        except Exception as e:
            print(
                f"  ⚠ Warning: Could not apply MONAI compatibility fixes: {e}")
            # Continue anyway - the model might still load

    def _manually_load_weights(self, bundle_path, device, already_loaded_files=None):
        """
        Manually load weights from bundle, bypassing ConfigParser's buggy weight loading.

        This works for ALL model types: UNet, DiNTS, SwinUNETR, etc.

        Args:
            bundle_path: Path to MONAI bundle directory
            device: Target device (cpu/cuda)
            already_loaded_files: Files that ConfigParser already tried to load

        Returns:
            bool: True if weights loaded successfully, False otherwise
        """
        models_dir = os.path.join(bundle_path, "models")

        if not os.path.exists(models_dir):
            print("  ⚠ No models directory found")
            return False

        # Try weight files in order of likelihood
        weight_candidates = [
            "model.pt",
            "model.pth",
            "checkpoint.pt",
            "best_model.pt",
            "model_weights.pt",
        ]

        for weight_file in weight_candidates:
            weight_path = os.path.join(models_dir, weight_file)

            if not os.path.exists(weight_path):
                continue

            print(f"  Trying {weight_file}...")

            try:
                # Load checkpoint
                checkpoint = torch.load(
                    weight_path, map_location=device, weights_only=False)

                # Extract state dict from various checkpoint formats
                state_dict = None

                if isinstance(checkpoint, dict):
                    # Try common state dict keys
                    for key in ['state_dict', 'model_state_dict', 'model', 'network']:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            print(
                                f"    Found state_dict in checkpoint['{key}']")
                            break

                    # If no known key, checkpoint itself might be the state dict
                    if state_dict is None:
                        # Check if it looks like a state dict (has weight/bias keys)
                        if any('weight' in k or 'bias' in k for k in checkpoint.keys()):
                            state_dict = checkpoint
                            print(f"    Checkpoint is the state_dict")

                elif isinstance(checkpoint, torch.nn.Module):
                    # Full model saved
                    state_dict = checkpoint.state_dict()
                    print(f"    Extracted state_dict from full model")

                if state_dict is None:
                    print(f"    ⚠ Could not find state_dict in {weight_file}")
                    continue

                # Try to load state dict into model
                try:
                    # Strict loading (all keys must match)
                    self.model.load_state_dict(state_dict, strict=True)
                    print(f"  ✓ Loaded weights from {weight_file} (strict)")

                    # Verify weights are not all zeros
                    first_param = next(self.model.parameters()).flatten()[:10]
                    if torch.all(first_param == 0):
                        print(f"    ⚠ Warning: Parameters are all zeros!")
                        continue

                    print(f"    Sample weights: {first_param[:5]}")
                    return True

                except RuntimeError as e:
                    if "size mismatch" in str(e) or "Missing key" in str(e):
                        print(f"    ⚠ Strict loading failed: {e}")

                        # Try non-strict loading
                        try:
                            missing, unexpected = self.model.load_state_dict(
                                state_dict, strict=False)
                            print(
                                f"  ✓ Loaded weights from {weight_file} (non-strict)")

                            if missing:
                                print(f"    ⚠ Missing keys: {len(missing)}")
                            if unexpected:
                                print(
                                    f"    ⚠ Unexpected keys: {len(unexpected)}")

                            # Verify weights are loaded
                            first_param = next(
                                self.model.parameters()).flatten()[:10]
                            if torch.all(first_param == 0):
                                print(f"    ⚠ Warning: Parameters are all zeros!")
                                continue

                            print(f"    Sample weights: {first_param[:5]}")
                            return True

                        except Exception as e2:
                            print(
                                f"    ⚠ Non-strict loading also failed: {e2}")
                            continue
                    else:
                        print(f"    ⚠ Loading failed: {e}")
                        continue

            except Exception as e:
                print(f"    ⚠ Error loading {weight_file}: {e}")
                continue

        # FALLBACK: Search entire bundle directory for .pt/.pth files
        print(f"  ⚠ Standard weight files not found, searching entire bundle...")

        found_weights = []
        for root, dirs, files in os.walk(bundle_path):
            for file in files:
                if file.endswith(('.pt', '.pth')) and not file.startswith('.'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, bundle_path)
                    found_weights.append((file, full_path, rel_path))

        if not found_weights:
            print(f"  ⚠ No .pt or .pth files found in entire bundle")
            return False

        print(f"  Found {len(found_weights)} weight file(s):")
        for fname, fpath, rel_path in found_weights:
            file_size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"    - {rel_path} ({file_size_mb:.1f} MB)")

        # Try each found weight file
        for fname, weight_path, rel_path in found_weights:
            print(f"  Trying {rel_path}...")

            try:
                # Load checkpoint
                checkpoint = torch.load(
                    weight_path, map_location=device, weights_only=False)

                # Extract state dict from various checkpoint formats
                state_dict = None

                if isinstance(checkpoint, dict):
                    # Try common state dict keys
                    for key in ['state_dict', 'model_state_dict', 'model', 'network']:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            print(
                                f"    Found state_dict in checkpoint['{key}']")
                            break

                    # If no known key, checkpoint itself might be the state dict
                    if state_dict is None:
                        # Check if it looks like a state dict (has weight/bias keys)
                        if any('weight' in k or 'bias' in k for k in checkpoint.keys()):
                            state_dict = checkpoint
                            print(f"    Checkpoint is the state_dict")

                elif isinstance(checkpoint, torch.nn.Module):
                    # Full model saved
                    state_dict = checkpoint.state_dict()
                    print(f"    Extracted state_dict from full model")

                if state_dict is None:
                    print(f"    ⚠ Could not find state_dict in {fname}")
                    continue

                # Try to load state dict into model
                try:
                    # Strict loading (all keys must match)
                    self.model.load_state_dict(state_dict, strict=True)
                    print(f"  ✓ Loaded weights from {rel_path} (strict)")

                    # Verify weights are not all zeros
                    first_param = next(self.model.parameters()).flatten()[:10]
                    if torch.all(first_param == 0):
                        print(f"    ⚠ Warning: Parameters are all zeros!")
                        continue

                    print(f"    Sample weights: {first_param[:5]}")
                    return True

                except RuntimeError as e:
                    if "size mismatch" in str(e) or "Missing key" in str(e):
                        print(
                            f"    ⚠ Strict loading failed: {str(e)[:100]}...")

                        # Try non-strict loading
                        try:
                            missing, unexpected = self.model.load_state_dict(
                                state_dict, strict=False)
                            print(
                                f"  ✓ Loaded weights from {rel_path} (non-strict)")

                            if missing:
                                print(f"    ⚠ Missing keys: {len(missing)}")
                            if unexpected:
                                print(
                                    f"    ⚠ Unexpected keys: {len(unexpected)}")

                            # Verify weights are loaded
                            first_param = next(
                                self.model.parameters()).flatten()[:10]
                            if torch.all(first_param == 0):
                                print(f"    ⚠ Warning: Parameters are all zeros!")
                                continue

                            print(f"    Sample weights: {first_param[:5]}")
                            return True

                        except Exception as e2:
                            print(
                                f"    ⚠ Non-strict loading also failed: {e2}")
                            continue
                    else:
                        print(f"    ⚠ Loading failed: {e}")
                        continue

            except Exception as e:
                print(f"    ⚠ Error loading {fname}: {e}")
                continue

        print(f"  ⚠ Could not load weights from any found .pt/.pth file")
        return False

    def _override_all_devices_in_config(self, parser, device_str):
        """
        Recursively override ALL device references in the config.

        This is critical for models like DiNTS that have 'cuda' hardcoded.
        """
        def recursive_override(obj):
            """Recursively traverse and override device values."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'device':
                        # Override device
                        obj[key] = device_str
                    elif isinstance(value, (dict, list)):
                        recursive_override(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        recursive_override(item)

        # Override in the raw config dictionary
        if hasattr(parser, 'config') and isinstance(parser.config, dict):
            recursive_override(parser.config)

        # Also set the top-level device
        parser['device'] = device_str

        # Override in specific known locations for DiNTS
        try:
            if 'network_def' in parser.config:
                network_def = parser.config['network_def']
                if isinstance(network_def, dict) and 'device' in network_def:
                    network_def['device'] = device_str

            if 'network' in parser.config:
                network = parser.config['network']
                if isinstance(network, dict) and 'device' in network:
                    network['device'] = device_str
        except:
            pass

        """
        Verify that loaded weights are correct by comparing with direct load.
        """
        try:
            # Find the weights file
            weights_candidates = [
                os.path.join(bundle_path, "models", "model.pt"),
                os.path.join(bundle_path, "models", "model.pth"),
                os.path.join(bundle_path, "models", "model.ts"),
            ]

            weights_file = None
            for candidate in weights_candidates:
                if os.path.exists(candidate):
                    weights_file = candidate
                    break

            if weights_file is None:
                print("  No weights file found for verification")
                return True

            # Load weights directly
            checkpoint = torch.load(
                weights_file, map_location=device, weights_only=False)

            # Get state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    direct_state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    direct_state_dict = checkpoint['model_state_dict']
                else:
                    direct_state_dict = checkpoint
            else:
                direct_state_dict = checkpoint

            # Compare first parameter
            model_params = dict(self.model.named_parameters())
            first_param_name = list(model_params.keys())[0]

            model_first_param = model_params[first_param_name].flatten()[:10]

            # Handle potential prefix differences
            if first_param_name in direct_state_dict:
                direct_first_param = direct_state_dict[first_param_name]
            elif 'model.' + first_param_name in direct_state_dict:
                direct_first_param = direct_state_dict['model.' +
                                                       first_param_name]
            elif first_param_name.replace('model.', '') in direct_state_dict:
                direct_first_param = direct_state_dict[first_param_name.replace(
                    'model.', '')]
            else:
                print(f"  Could not find matching param in checkpoint")
                return True

            direct_first_param = direct_first_param.flatten()[:10].to(device)

            # Compare
            match = torch.allclose(
                model_first_param, direct_first_param, rtol=1e-4)

            return match

        except Exception as e:
            print(f"  Verification error: {e}")
            return True

    def _manually_load_monai_model(self, bundle_path, parser, device):
        """
        Manually load MONAI model by detecting architecture and loading weights.
        """
        try:
            model_type = self._detect_monai_model_architecture(parser)

            print(f"  Detected architecture: {model_type}")

            if model_type == "UNet":
                return self._load_unet_manually(bundle_path, parser, device)
            elif model_type == "DiNTS":
                print("  ⚠ DiNTS requires config loading (too complex for manual)")
                return None
            elif model_type == "SwinUNETR":
                return self._load_swinunetr_manually(bundle_path, parser, device)
            else:
                print(f"  Unknown model type: {model_type}")
                return None

        except Exception as e:
            print(f"  Manual loading failed: {e}")
            return None

    def _detect_monai_model_architecture(self, parser):
        """Detect model architecture from bundle config."""
        try:
            config = parser.config if hasattr(parser, 'config') else {}
            network_config = config.get(
                'network_def', config.get('network', {}))

            if isinstance(network_config, dict):
                target = network_config.get('_target_', '')

                # Check _target_ string
                if 'TopologyInstance' in target or 'DiNTS' in target:
                    return "DiNTS"
                elif 'SwinUNETR' in target:
                    return "SwinUNETR"
                elif 'SegResNet' in target:
                    return "SegResNet"
                elif 'UNet' in target and 'Swin' not in target:
                    return "UNet"

                # Also check #network_def reference
                if network_config.get('#network_def'):
                    ref = network_config['#network_def']
                    if 'dints' in str(ref).lower():
                        return "DiNTS"

            # Fallback: check bundle path name
            bundle_name = os.path.basename(os.path.dirname(
                config_file) if 'config_file' in locals() else '').lower()

            if 'dints' in bundle_name or 'pancreas' in bundle_name:
                return "DiNTS"
            elif 'swin' in bundle_name:
                return "SwinUNETR"
            elif 'spleen' in bundle_name or 'liver' in bundle_name:
                return "UNet"

            return "Unknown"

        except Exception as e:
            print(f"  Detection error: {e}")
            return "Unknown"

        """Manually load SwinUNETR architecture."""
        try:
            from monai.networks.nets import SwinUNETR

            print("  Creating SwinUNETR model...")

            # Try to extract parameters from config
            try:
                network_config = parser.config.get(
                    'network_def', parser.config.get('network', {}))

                img_size = network_config.get('img_size', (96, 96, 96))
                in_channels = network_config.get('in_channels', 1)
                out_channels = network_config.get('out_channels', 2)
                feature_size = network_config.get('feature_size', 48)

            except:
                # Defaults
                img_size = (96, 96, 96)
                in_channels = 1
                out_channels = 2
                feature_size = 48

            model = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
            )

            # Load weights
            weights_file = os.path.join(bundle_path, "models", "model.pt")
            if os.path.exists(weights_file):
                checkpoint = torch.load(
                    weights_file, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()

            return model

        except ImportError:
            raise ImportError(
                "SwinUNETR requires additional dependencies. "
                "Install with: pip install 'monai[all]'"
            )

    def _load_keras_json(self, model_path):
        """Load Keras model from JSON architecture file."""
        handler = _CustomObjectHandler()
        custom_objects = handler.prepare_custom_objects(model_path)

        with open(model_path, "r") as json_file:
            json_content = json_file.read()

        self.model = tf.keras.models.model_from_json(
            json_content, custom_objects=custom_objects)
        self.model_type = 'keras'

        # Optionally load weights
        weights_path, _ = QFileDialog.getOpenFileName(
            None, "Load Model Weights (Optional)",
            os.path.dirname(model_path), "Weight Files (*.h5 *.weights.h5);;All Files (*.*)")

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)

        self.input_dims = self._infer_keras_dims(self.model)
        return True

    def _load_keras_yaml(self, model_path):
        """Load Keras model from YAML architecture file."""
        with open(model_path, "r") as yaml_file:
            yaml_content = yaml_file.read()

        self.model = tf.keras.models.model_from_yaml(yaml_content)
        self.model_type = 'keras'

        # Optionally load weights
        weights_path, _ = QFileDialog.getOpenFileName(
            None, "Load Model Weights (Optional)",
            os.path.dirname(model_path), "Weight Files (*.h5 *.weights.h5);;All Files (*.*)")

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)

        self.input_dims = self._infer_keras_dims(self.model)
        return True

    def _load_keras_h5(self, model_path):
        """Load complete Keras model from H5 file."""
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model_type = 'keras'
            self.input_dims = self._infer_keras_dims(self.model)
            return True

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Keras H5 model: {model_path}\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

    def _load_pytorch_ts(self, model_path, weights_path=None):
        """Load TorchScript model with optional external weights.

        Args:
            model_path: Path to TorchScript (.ts) file
            weights_path: Optional path to separate weights file (.pth, .pt)
        """
        device = self._get_torch_device()

        # Load the TorchScript model architecture
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.to(device)

        # If weights_path is provided, load and apply external weights
        if weights_path:
            print(f"Loading external weights from: {weights_path}")

            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights file not found: {weights_path}")

            try:
                # Load the weights
                weights = torch.load(
                    weights_path, map_location=device, weights_only=False)

                # Handle different weight formats
                if isinstance(weights, dict):
                    if 'state_dict' in weights:
                        state_dict = weights['state_dict']
                    elif 'model_state_dict' in weights:
                        state_dict = weights['model_state_dict']
                    elif 'model' in weights:
                        # If 'model' contains a state_dict
                        if isinstance(weights['model'], dict):
                            state_dict = weights['model']
                        else:
                            # 'model' is a full model, extract state_dict
                            state_dict = weights['model'].state_dict()
                    else:
                        # Assume the dict itself is the state_dict
                        state_dict = weights
                else:
                    # If weights is a model object, extract state_dict
                    if hasattr(weights, 'state_dict'):
                        state_dict = weights.state_dict()
                    else:
                        raise ValueError("Unsupported weights format")

                # Load state dict into the model
                self.model.load_state_dict(state_dict, strict=False)
                print(f"✓ External weights loaded successfully")

            except Exception as e:
                print(f"⚠ Warning: Failed to load external weights: {e}")
                print(f"  Using weights embedded in TorchScript file")

        self.model.eval()
        self.model_type = 'pytorch'
        self.input_dims = self._torch_find_input_dim(self.model)
        return True

    def _load_pytorch_checkpoint(self, model_path):
        """Load PyTorch checkpoint."""
        device = self._get_torch_device()
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model = checkpoint['model']
        else:
            raise ValueError("PyTorch checkpoint format not supported. "
                             "Need complete model or 'model' key in checkpoint.")

        self.model.to(device)
        self.model.eval()
        self.model_type = 'pytorch'
        self.input_dims = self._torch_find_input_dim(self.model)
        return True

    def get_prediction(self, input_data, spacing=None, origin=None, direction=None):
        """Universal prediction method."""
        if self.model_type == "monai":
            return self._predict_monai(input_data, spacing, origin, direction)
        elif self.model_type == "nnunet":
            if spacing is None:
                raise ValueError("Spacing required for nnU-Net")
            return self.predict_nnunet(input_data, spacing, origin, direction)
        elif self.input_dims == "2D":
            return self.predict_25D(input_data)
        elif self.input_dims == "3D":
            return self.predict_3D(input_data)

    def _predict_monai(self, input_data, spacing, origin, direction):
        """MONAI prediction with FULL debug output."""
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
            Spacingd, ScaleIntensityRanged, CropForegroundd, EnsureTyped,
            Invertd, AsDiscreted
        )
        from monai.data import Dataset, DataLoader, decollate_batch
        import tempfile
        import os

        device = self._get_torch_device()

        # Prepare data
        raw_image = input_data.copy() if isinstance(
            input_data, np.ndarray) else input_data
        spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        origin = origin if origin is not None else (0.0, 0.0, 0.0)
        direction = direction if direction is not None else np.eye(3)

        print(f"\n{'='*80}")
        print("MONAI PREDICTION - FULL DEBUG")
        print(f"{'='*80}")
        print(f"Input shape: {raw_image.shape}")
        print(
            f"Input HU range: [{raw_image.min():.1f}, {raw_image.max():.1f}]")
        print(f"Input dtype: {raw_image.dtype}")
        print(f"Input spacing: {spacing}")

        # Build affine
        affine = np.eye(4)
        affine[:3, :3] = np.array(direction) * np.array(spacing)
        affine[:3, 3] = origin

        # Save as temporary file
        temp_file = os.path.join(tempfile.gettempdir(),
                                 f"monai_temp_{os.getpid()}.nii.gz")

        try:
            # Save temp NIfTI
            nib.save(nib.Nifti1Image(raw_image, affine), temp_file)
            print(f"\n✓ Saved temp file: {temp_file}")

            # Create transforms
            pre_transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                EnsureTyped(keys=["image"], device=str(
                    device), track_meta=True),
            ])

            post_transforms = Compose([
                EnsureTyped(keys="pred", device=str(device)),
                Invertd(
                    keys="pred", transform=pre_transforms, orig_keys="image",
                    meta_keys="pred_meta_dict", orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict", nearest_interp=False, to_tensor=True,
                ),
                AsDiscreted(keys="pred", argmax=True),
            ])

            print("\n" + "="*80)
            print("STEP 1: PREPROCESSING")
            print("="*80)

            # Create dataset
            data = [{"image": temp_file}]
            dataset = Dataset(data=data, transform=pre_transforms)
            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

            # Get preprocessed data
            for batch_data in dataloader:
                inputs = batch_data["image"].to(device)

                print(f"✓ Preprocessed image:")
                print(f"  Shape: {inputs.shape}")
                print(f"  Dtype: {inputs.dtype}")
                print(f"  Device: {inputs.device}")
                print(f"  Range: [{inputs.min():.4f}, {inputs.max():.4f}]")
                print(f"  Mean: {inputs.mean():.4f}")
                print(f"  Std: {inputs.std():.4f}")

                # Check if preprocessing looks correct
                if inputs.min() < -0.1 or inputs.max() > 1.1:
                    print(f"\n  ⚠ WARNING: Values not in [0,1] range!")

                print("\n" + "="*80)
                print("STEP 2: MODEL INFERENCE")
                print("="*80)

                # Check model state
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"Model training: {self.model.training}")

                # Inference
                with torch.no_grad():
                    prediction = sliding_window_inference(
                        inputs=inputs,
                        roi_size=(96, 96, 96),
                        sw_batch_size=1,
                        predictor=self.model,
                        overlap=0.5,
                        progress=False
                    )

                print(f"\n✓ Raw model output:")
                print(f"  Shape: {prediction.shape}")
                print(f"  Dtype: {prediction.dtype}")
                print(
                    f"  Range: [{prediction.min():.4f}, {prediction.max():.4f}]")

                # CRITICAL: Check logits distribution
                pred_no_batch = prediction.squeeze(0)  # [C, H, W, D]
                if pred_no_batch.shape[0] == 2:
                    class_0 = pred_no_batch[0]
                    class_1 = pred_no_batch[1]

                    print(f"\n  Logits analysis:")
                    print(
                        f"    Class 0 (BG):  min={class_0.min():.4f}, max={class_0.max():.4f}, mean={class_0.mean():.4f}, std={class_0.std():.4f}")
                    print(
                        f"    Class 1 (FG):  min={class_1.min():.4f}, max={class_1.max():.4f}, mean={class_1.mean():.4f}, std={class_1.std():.4f}")

                    # Check decision boundary
                    diff = class_1 - class_0
                    print(
                        f"    Difference:    min={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}")

                    # Count predictions
                    fg_wins = (class_1 > class_0).sum().item()
                    total = class_0.numel()
                    fg_pct = 100 * fg_wins / total

                    print(f"\n  Prediction analysis:")
                    print(
                        f"    Foreground wins: {fg_wins:,} / {total:,} ({fg_pct:.2f}%)")

                    # Sample some voxels
                    print(f"\n  Sample voxels (first 10):")
                    flat_0 = class_0.flatten()[:10].cpu().numpy()
                    flat_1 = class_1.flatten()[:10].cpu().numpy()
                    for i in range(10):
                        winner = "FG" if flat_1[i] > flat_0[i] else "BG"
                        print(
                            f"    Voxel {i}: BG={flat_0[i]:.4f}, FG={flat_1[i]:.4f} → {winner}")

                    # CRITICAL CHECK
                    if fg_pct > 50:
                        print(
                            f"\n  ⚠⚠⚠ CRITICAL ISSUE: {fg_pct:.1f}% classified as foreground!")
                        print(f"  Expected: ~2-5% for spleen")
                        print(f"  This means:")
                        print(f"    1. Model weights are wrong/corrupted")
                        print(f"    2. Preprocessing is incorrect")
                        print(
                            f"    3. Model architecture mismatch (despite same param count)")
                    elif fg_pct < 0.5:
                        print(
                            f"\n  ⚠ WARNING: Only {fg_pct:.3f}% as foreground - seems too low")
                    else:
                        print(f"\n  ✓ Foreground percentage looks reasonable")

                print("\n" + "="*80)
                print("STEP 3: POSTPROCESSING")
                print("="*80)

                # Store prediction
                batch_data["pred"] = prediction

                # Decollate
                print("Decollating batch...")
                batch_list = decollate_batch(batch_data)
                print(f"  Batch list length: {len(batch_list)}")
                print(f"  Keys in item 0: {batch_list[0].keys()}")

                # Apply postprocessing
                print("\nApplying postprocessing (Invertd + AsDiscreted)...")
                try:
                    processed = post_transforms(batch_list[0])
                    print("  ✓ Postprocessing completed")
                    print(f"  Keys in processed: {processed.keys()}")
                except Exception as e:
                    print(f"  ✗ Postprocessing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

                # Extract result
                final_pred = processed["pred"]

                print(f"\n✓ Final prediction:")
                print(f"  Type: {type(final_pred)}")
                if isinstance(final_pred, torch.Tensor):
                    print(f"  Shape: {final_pred.shape}")
                    print(f"  Dtype: {final_pred.dtype}")
                    print(f"  Device: {final_pred.device}")
                    result = final_pred.cpu().numpy()
                else:
                    print(f"  Shape: {final_pred.shape}")
                    print(f"  Dtype: {final_pred.dtype}")
                    result = np.array(final_pred)

                result = result.astype(np.uint8)

                # Remove extra dimensions
                while result.ndim > 3:
                    result = result.squeeze()

                # Final analysis
                unique = np.unique(result)
                fg_voxels = np.sum(result > 0)
                total_voxels = result.size
                fg_pct_final = 100 * fg_voxels / total_voxels

                print(f"\n" + "="*80)
                print("FINAL RESULT")
                print("="*80)
                print(f"Shape: {result.shape}")
                print(f"Unique classes: {unique}")
                print(
                    f"Foreground voxels: {fg_voxels:,} / {total_voxels:,} ({fg_pct_final:.2f}%)")

                if fg_pct_final > 50:
                    print(f"\n⚠⚠⚠ STILL WRONG after postprocessing!")
                    print(f"This suggests the MODEL ITSELF is broken.")
                    print(f"\nDEBUG ACTIONS:")
                    print(f"1. Check if the bundle weights are actually loaded:")
                    print(f"   - Look at bundle_path/models/model.pt")
                    print(f"   - Verify it's the correct file")
                    print(f"2. Try loading model.pt directly into the UNet:")
                    print(f"   model = UNet(...)")
                    print(f"   model.load_state_dict(torch.load('models/model.pt'))")
                    print(f"   Then test with that model in your app")
                elif fg_pct_final < 1:
                    print(
                        f"\n⚠ Very few foreground voxels - check if spleen is actually visible")
                else:
                    print(f"\n✓ Result looks reasonable!")

                print("="*80 + "\n")

                return result

        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def predict_3D(self, input_data):
        """3D prediction for Keras/PyTorch models."""
        if self.model_type == "keras":
            return self._predict_3D_keras(input_data)
        elif self.model_type == "pytorch":
            return self._predict_3D_pytorch(input_data)

    def _predict_3D_keras(self, input_data):
        """Keras 3D prediction."""
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.expand_dims(input_data, axis=-1)

        prediction = self.model.predict(input_data, verbose=0)
        prediction = prediction.squeeze()

        if len(prediction.shape) > 3:
            if prediction.shape[-1] > 1:
                prediction = np.argmax(prediction, axis=-1)
            else:
                prediction = np.round(prediction)

        return prediction.astype(np.uint8)

    def _predict_3D_pytorch(self, input_data):
        """PyTorch 3D prediction."""
        device = self._get_torch_device()

        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data

        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.ndim == 4:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(device)
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            prediction = self.model(input_tensor)

        prediction = prediction.squeeze()

        if prediction.ndim == 4:
            if prediction.shape[0] > 1:
                prediction = torch.argmax(prediction, dim=0)
            else:
                prediction = torch.round(prediction)

        return prediction.cpu().numpy().astype(np.uint8)

    def predict_25D(self, input_data):
        """2.5D prediction (slice-by-slice)."""
        # Implementation omitted for brevity - same as before
        pass

    def _get_torch_device(self):
        """Get PyTorch device based on settings."""
        if self.processing_type == "gpu" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _infer_keras_dims(self, model):
        """Infer if Keras model is 2D or 3D."""
        has_conv3d = any(isinstance(layer, tf.keras.layers.Conv3D)
                         for layer in model.layers)
        has_conv2d = any(isinstance(layer, tf.keras.layers.Conv2D)
                         for layer in model.layers)
        return "3D" if has_conv3d else "2D"

    def _torch_find_input_dim(self, module):
        """
        Find input dimensionality of PyTorch model.

        Improved version that:
        1. Handles TorchScript models better
        2. Uses test forward pass as fallback
        3. Doesn't blindly default to 3D
        4. Returns None if uncertain
        """

        # Method 1: Check layer names (works for regular PyTorch models)
        dim_from_layers = self._check_layer_names(module)
        if dim_from_layers:
            print(f"Detected {dim_from_layers} from layer names")
            return dim_from_layers

        # Method 2: Test with dummy inputs (works for TorchScript)
        dim_from_test = self._test_input_dimensions(module)
        if dim_from_test:
            print(f"Detected {dim_from_test} from test forward pass")
            return dim_from_test

        # Method 3: Check if it's TorchScript and inspect code
        if isinstance(module, torch.jit.ScriptModule):
            dim_from_code = self._check_torchscript_code(module)
            if dim_from_code:
                print(f"Detected {dim_from_code} from TorchScript code")
                return dim_from_code

        # If all methods fail, return None (will trigger manual input)
        print("WARNING: Could not detect model dimensionality")
        return None

    def _check_layer_names(self, module):
        """Method 1: Check layer names recursively."""
        if hasattr(module, "_get_name"):
            name = module._get_name()
            if "Conv2d" in name:
                return "2D"
            elif "Conv3d" in name:
                return "3D"

        for child in module.children():
            dim = self._check_layer_names(child)
            if dim:
                return dim

        return None

    def _test_input_dimensions(self, module):
        """
        Method 2: Test with dummy inputs to determine dimensionality.
        Try 2D first, then 3D.
        """
        device = self._get_torch_device()
        module.to(device)
        module.eval()

        # Test 2D input [Batch, Channel, H, W]
        try:
            with torch.no_grad():
                test_2d = torch.randn(1, 1, 64, 64).to(device)
                output = module(test_2d)

                # Check if output has reasonable shape for 2D
                if output.dim() in [3, 4]:  # [B, H, W] or [B, C, H, W]
                    print(
                        f"  2D test successful: input {test_2d.shape} -> output {output.shape}")
                    return "2D"
        except Exception as e:
            error_msg = str(e).lower()
            # If it explicitly expects 3D/5D, it's not a 2D model
            if "expected 3d" in error_msg or "expected 5d" in error_msg:
                print(f"  2D test failed (expected): {e}")
                pass
            else:
                print(f"  2D test failed: {e}")

        # Test 3D input [Batch, Channel, D, H, W]
        try:
            with torch.no_grad():
                test_3d = torch.randn(1, 1, 32, 64, 64).to(device)
                output = module(test_3d)

                # Check if output has reasonable shape for 3D
                if output.dim() in [4, 5]:  # [B, D, H, W] or [B, C, D, H, W]
                    print(
                        f"  3D test successful: input {test_3d.shape} -> output {output.shape}")
                    return "3D"
        except Exception as e:
            print(f"  3D test failed: {e}")

        return None

    def _check_torchscript_code(self, module):
        """
        Method 3: Inspect TorchScript code for conv operations.
        """
        try:
            # Get the TorchScript code
            code = module.code
            code_str = str(code)

            # Look for conv operations in the code
            if "conv2d" in code_str.lower():
                return "2D"
            elif "conv3d" in code_str.lower():
                return "3D"

            # Check for F.conv2d or F.conv3d
            if "torch.conv2d" in code_str or "F.conv2d" in code_str:
                return "2D"
            elif "torch.conv3d" in code_str or "F.conv3d" in code_str:
                return "3D"

        except Exception as e:
            print(f"  TorchScript code inspection failed: {e}")

        return None

    def check_model_compatibility(self, image_shape):
        """
        Check if image is compatible with model.

        Args:
            image_shape (tuple): Shape of the input image (D, H, W) or (H, W)

        Returns:
            tuple: (is_compatible, needed_size, mode)
                - is_compatible (bool): True if image matches model input
                - needed_size (tuple or None): Required size if not compatible
                - mode (str or None): Processing mode ('pad', 'resize', etc.)
        """
        if self.model is None:
            return False, None, None

        # For MONAI models, we can try to infer from config
        if self.model_type == "monai":
            # MONAI models often handle arbitrary sizes with sliding window
            # Return compatible by default, let MONAI handle it
            return True, None, None

        # Try to get input shape from model
        try:
            if self.model_type == "keras":
                return self._check_keras_compatibility(image_shape)
            elif self.model_type == "pytorch":
                return self._check_pytorch_compatibility(image_shape)
        except Exception as e:
            print(f"Error checking model compatibility: {e}")
            # If we can't determine, assume incompatible and return None
            # This will trigger the manual input dialog
            return False, None, None

        # Default: assume compatible
        return True, None, None

    def _check_keras_compatibility(self, image_shape):
        """Check compatibility for Keras models."""
        try:
            # Get the input shape from the model
            if not hasattr(self.model, 'input_shape'):
                print("Warning: Keras model has no input_shape attribute")
                return False, None, None

            input_shape = self.model.input_shape

            if input_shape is None:
                print("Warning: Keras model input_shape is None")
                return False, None, None

            # Remove batch dimension
            if isinstance(input_shape, list):
                if len(input_shape) == 0:
                    print("Warning: Keras model input_shape list is empty")
                    return False, None, None
                input_shape = input_shape[0]

            # Skip batch dimension (first element)
            if len(input_shape) <= 1:
                print(
                    f"Warning: Keras model input_shape too short: {input_shape}")
                return False, None, None

            model_shape = input_shape[1:]  # Skip batch dimension

            print(
                f"Debug: Keras model shape (after removing batch): {model_shape}")
            print(f"Debug: Image shape: {image_shape}")
            print(f"Debug: Input channels setting: {self.input_channels}")

            # Handle None dimensions (flexible dimensions)
            # If model has None dimensions, we consider it flexible/compatible
            if None in model_shape:
                print(
                    "Debug: Model has flexible dimensions (None), considering compatible")
                return True, None, None

            # Auto-detect channel format if not set
            # Convention: channel dimension is typically 1 or 3 (RGB)
            if self.input_channels is None:
                if len(model_shape) == 3:  # 2D model
                    # Check if first or last dimension is likely channels
                    if model_shape[0] in [1, 3, 4]:
                        self.input_channels = "channel_first"
                        print(
                            f"Debug: Auto-detected channel_first (first dim = {model_shape[0]})")
                    elif model_shape[-1] in [1, 3, 4]:
                        self.input_channels = "channel_last"
                        print(
                            f"Debug: Auto-detected channel_last (last dim = {model_shape[-1]})")
                    else:
                        # Ambiguous, default to channel_last (TensorFlow/Keras convention)
                        self.input_channels = "channel_last"
                        print("Debug: Ambiguous format, defaulting to channel_last")

                elif len(model_shape) == 4:  # 3D model
                    # Check if first or last dimension is likely channels
                    if model_shape[0] in [1, 3, 4]:
                        self.input_channels = "channel_first"
                        print(
                            f"Debug: Auto-detected channel_first (first dim = {model_shape[0]})")
                    elif model_shape[-1] in [1, 3, 4]:
                        self.input_channels = "channel_last"
                        print(
                            f"Debug: Auto-detected channel_last (last dim = {model_shape[-1]})")
                    else:
                        # Ambiguous, default to channel_last (TensorFlow/Keras convention)
                        self.input_channels = "channel_last"
                        print("Debug: Ambiguous format, defaulting to channel_last")

            # Determine if 2D or 3D based on model_shape length
            if len(model_shape) == 3:  # (H, W, C) or (C, H, W)
                # 2D model
                if len(image_shape) == 3:
                    # Image is 3D, not compatible with 2D model
                    print("Debug: 2D model but 3D image - incompatible")
                    return False, None, None

                # Extract spatial dimensions based on channel format
                if self.input_channels == "channel_last":
                    required = tuple(model_shape[:2])  # (H, W) - first 2 dims
                elif self.input_channels == "channel_first":
                    required = tuple(model_shape[1:])  # (H, W) - last 2 dims
                else:
                    # No channels, all spatial
                    required = tuple(model_shape)

                print(f"Debug: Required shape for 2D: {required}")

                if tuple(image_shape) == required:
                    return True, None, None
                else:
                    return False, required, "resize"

            elif len(model_shape) == 4:  # (D, H, W, C) or (C, D, H, W)
                # 3D model
                if len(image_shape) == 2:
                    # Image is 2D, not compatible with 3D model
                    print("Debug: 3D model but 2D image - incompatible")
                    return False, None, None

                # Extract spatial dimensions based on channel format
                if self.input_channels == "channel_last":
                    # (D, H, W) - first 3 dims
                    required = tuple(model_shape[:3])
                elif self.input_channels == "channel_first":
                    # (D, H, W) - last 3 dims
                    required = tuple(model_shape[1:])
                else:
                    # No channels, all spatial
                    required = tuple(model_shape)

                print(f"Debug: Required shape for 3D: {required}")
                print(f"Debug: Image shape: {tuple(image_shape)}")

                if tuple(image_shape) == required:
                    print("Debug: Shapes match! Compatible.")
                    return True, None, None
                else:
                    print("Debug: Shapes don't match. Incompatible.")
                    return False, required, "resize"

            # Unknown model shape
            print(
                f"Warning: Unknown Keras model shape length: {len(model_shape)}")
            return False, None, None

        except AttributeError as e:
            print(f"Keras compatibility check failed - AttributeError: {e}")
            print("This might be a custom model without standard attributes")
            return False, None, None
        except Exception as e:
            print(f"Keras compatibility check failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None, None

    def _check_pytorch_compatibility(self, image_shape):
        """Check compatibility for PyTorch models."""
        try:
            # Try to infer input shape by doing a forward pass with dummy data
            device = self._get_torch_device()
            self.model.to(device)
            self.model.eval()

            # Determine dimensionality from image shape
            if len(image_shape) == 2:
                # 2D image
                test_input = torch.randn(1, 1, *image_shape).to(device)
            else:
                # 3D image
                test_input = torch.randn(1, 1, *image_shape).to(device)

            # Try forward pass
            try:
                with torch.no_grad():
                    output = self.model(test_input)

                # If it works, shapes are compatible
                return True, None, None

            except RuntimeError as e:
                error_msg = str(e)

                # Check if it's a dimension mismatch error
                if "Expected" in error_msg and "input" in error_msg.lower():
                    # Try to extract expected dimensions from error
                    # This is tricky and may not always work

                    # Check if it's a 2D/3D mismatch
                    if "Expected 3D" in error_msg or "Expected 4D" in error_msg:
                        if "conv2d" in error_msg.lower():
                            # Model is 2D, image is 3D
                            print("Model is 2D but image is 3D")
                            return False, None, None
                        elif "conv3d" in error_msg.lower():
                            # Model is 3D, image is 2D
                            print("Model is 3D but image is 2D")
                            return False, None, None

                    # Can't determine exact size needed
                    print(f"Dimension mismatch detected: {error_msg}")
                    return False, None, None

                # Some other error
                print(f"Model forward pass failed: {error_msg}")
                return False, None, None

        except Exception as e:
            print(f"PyTorch compatibility check failed: {e}")
            # If check fails, assume incompatible with unknown size
            return False, None, None

    def predict_nnunet(self, image_data, spacing, origin=None, direction=None):
        """
        Run nnU-Net prediction (works for both V1 and V2).

        Args:
            image_data: Input image array
            spacing: Voxel spacing tuple
            origin: Image origin (optional, defaults to (0,0,0))
            direction: Direction cosine matrix (optional, defaults to identity)

        Returns:
            Segmentation mask
        """
        # Route to appropriate version
        if self.nnunet_version == 2:
            if self.nnunet_handler is None:
                raise ValueError("nnU-Net V2 model not loaded")

            segmentation = self.nnunet_handler.predict(
                image_data,
                spacing,
                origin=origin,
                direction=direction
            )
        elif self.nnunet_version == 1:
            if self.nnunet_v1_handler is None:
                raise ValueError("nnU-Net V1 model not loaded")

            segmentation = self.nnunet_v1_handler.predict(
                image_data,
                spacing,
                origin=origin,
                direction=direction
            )
        else:
            raise ValueError("No nnU-Net model loaded")

        if segmentation is None:
            raise ValueError("Prediction returned None!")

        return segmentation

    def get_nnunet_class_info(self):
        """Get class information from nnU-Net model (works for both V1 and V2)."""
        if self.nnunet_version == 2 and self.nnunet_handler is not None:
            return {
                'num_classes': self.nnunet_handler.get_num_classes(),
                'class_names': self.nnunet_handler.get_class_names(),
                'configuration': self.nnunet_handler.configuration,
                'patch_size': self.nnunet_handler.get_patch_size(),
                'version': 'V2'
            }
        elif self.nnunet_version == 1 and self.nnunet_v1_handler is not None:
            patch_size = self.nnunet_v1_handler.get_patch_size()
            return {
                'num_classes': self.nnunet_v1_handler.get_num_classes(),
                'class_names': self.nnunet_v1_handler.get_class_names(),
                'configuration': self.nnunet_v1_handler.configuration,
                'patch_size': patch_size if patch_size else 'N/A',
                'version': 'V1'
            }

        return None


class _CustomObjectHandler:
    """Handler for custom Keras objects."""

    def __init__(self):
        self.custom_objects = {}

    def create_dummy_function(self, name):
        """Create dummy function for custom Keras object."""
        def dummy_func(y_true=None, y_pred=None, *args, **kwargs):
            if y_true is not None:
                return tf.constant(0.0)
            return None
        dummy_func.__name__ = name
        return dummy_func

    def extract_custom_names_from_json(self, json_path):
        """Extract custom function names from JSON."""
        with open(json_path, 'r') as f:
            config = json.load(f)

        custom_names = set()

        def recursive_search(obj):
            if isinstance(obj, dict):
                if 'class_name' in obj and obj.get('class_name') == 'function':
                    if 'config' in obj and isinstance(obj['config'], str):
                        custom_names.add(obj['config'])
                if 'registered_name' in obj and obj['registered_name'] not in ['None', None]:
                    if obj.get('class_name') == 'function':
                        custom_names.add(obj['registered_name'])
                for value in obj.values():
                    recursive_search(value)
            elif isinstance(obj, list):
                for item in obj:
                    recursive_search(item)

        recursive_search(config)
        return custom_names

    def prepare_custom_objects(self, json_path):
        """Prepare custom objects dictionary."""
        custom_names = self.extract_custom_names_from_json(json_path)
        for name in custom_names:
            self.custom_objects[name] = self.create_dummy_function(name)
        return self.custom_objects
