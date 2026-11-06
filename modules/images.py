import numpy as np
import nibabel as nib
import nrrd
from skimage.transform import resize
from PIL import Image as PILImage
import pydicom
from pydicom.pixels import apply_modality_lut
import os
from modules.dicomSeriesSelectionDialog import DicomSeriesSelectionDialog
from scipy import ndimage

class Image:
    def __init__(self, ww, wl):
        self.ww = ww
        self.wl = wl
        self.image_data = None
        self.processed_image = None
        self.pre_segmentation = None
        self.GT_segmentation = None
        self.spacing = None
        self.origin = None
        self.direction = None
        self.hight = None
        self.width = None
        self.slices = None
        self.num_of_class = None
        self.perm = None

    def get_ww(self):
        """
        Get the window width.
        :return: Window width.
        """
        return self.ww

    def get_wl(self):
        """
        Get the window level.
        :return: Window level.
        """
        return self.wl

    def load_image(self, image_path):
        """
        Load an image from a file.
        :param image_path: Path to the image file.
        """
        # Load the image using nibabel
        if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
            img = nib.load(image_path)
            self.image_data = img.get_fdata()
            affine = img.affine
            self.origin = affine[:3, 3]
            self.spacing = nib.affines.voxel_sizes(affine)

            # direction matrix = rotation/orientation
            self.direction = affine[:3, :3] / self.spacing

            # IMPORTANT: For MONAI models, DON'T force orientation here!
            # MONAI's Orientationd transform will handle orientation conversion
            # We just need to pass the correct affine matrix
            try:
                orientation = nib.aff2axcodes(img.affine)
                print(f"Original image orientation: {orientation}")
                # Store original orientation for reference
                self.original_orientation = orientation

                # Determine if we need to rotate views for proper medical display
                # Standard radiological orientation is RAS (Right-Anterior-Superior) or similar
                # We'll check if the axes need rotation for proper visualization
                self.needs_sagittal_rotation = self._check_sagittal_rotation_needed(
                    orientation)
                self.needs_coronal_rotation = self._check_coronal_rotation_needed(
                    orientation)

                print(
                    f"Sagittal rotation needed: {self.needs_sagittal_rotation}")
                print(
                    f"Coronal rotation needed: {self.needs_coronal_rotation}")

                # Don't transpose - keep data in original orientation
                # MONAI preprocessing will reorient if needed
            except Exception as e:
                print(f"Warning: Could not determine orientation: {e}")
                self.original_orientation = None
                self.needs_sagittal_rotation = False
                self.needs_coronal_rotation = False
                self.spacing = (1.0, 1.0, 1.0)
                self.origin = (0.0, 0.0, 0.0)
                self.direction = np.eye(3)

        elif image_path.endswith('.npy'):
            self.image_data = np.load(image_path)
            
            # Try to load metadata from sidecar JSON file if it exists
            json_path = image_path.replace('.npy', '.json')
            metadata_loaded = False
            
            if os.path.exists(json_path):
                try:
                    import json
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract spacing
                    if 'spacing' in metadata:
                        self.spacing = tuple(metadata['spacing'])
                    elif 'pixdim' in metadata:
                        self.spacing = tuple(metadata['pixdim'][:3])
                    else:
                        self.spacing = (1.0, 1.0, 1.0)
                    
                    # Extract origin
                    if 'origin' in metadata:
                        self.origin = tuple(metadata['origin'])
                    elif 'qoffset' in metadata:
                        self.origin = tuple(metadata['qoffset'])
                    else:
                        self.origin = (0.0, 0.0, 0.0)
                    
                    # Extract direction
                    if 'direction' in metadata:
                        direction = metadata['direction']
                        if isinstance(direction, list):
                            if len(direction) == 9:
                                self.direction = np.array(direction).reshape(3, 3)
                            elif len(direction) == 3 and all(isinstance(row, list) for row in direction):
                                self.direction = np.array(direction)
                            else:
                                self.direction = np.eye(3)
                        else:
                            self.direction = np.eye(3)
                    else:
                        self.direction = np.eye(3)
                    
                    # Extract orientation if available
                    if 'orientation' in metadata:
                        self.original_orientation = tuple(metadata['orientation'])
                        print(f"NPY file orientation (from metadata): {self.original_orientation}")
                        
                        # Determine rotation needs
                        self.needs_sagittal_rotation = self._check_sagittal_rotation_needed(
                            self.original_orientation)
                        self.needs_coronal_rotation = self._check_coronal_rotation_needed(
                            self.original_orientation)
                    else:
                        self.original_orientation = None
                        self.needs_sagittal_rotation = False
                        self.needs_coronal_rotation = False
                    
                    self.perm = None
                    metadata_loaded = True
                    print(f"✓ Loaded metadata from {json_path}")
                    print(f"  Spacing: {self.spacing}")
                    print(f"  Origin: {self.origin}")
                    print(f"  Direction:\n{self.direction}")
                    
                except Exception as e:
                    print(f"Warning: Could not load metadata from {json_path}: {e}")
                    print("  Using default values")
                    metadata_loaded = False
            
            # If no metadata file or loading failed, use defaults
            if not metadata_loaded:
                self.spacing = (1.0, 1.0, 1.0)
                self.origin = (0.0, 0.0, 0.0)
                self.direction = np.eye(3)
                self.original_orientation = None
                self.needs_sagittal_rotation = False
                self.needs_coronal_rotation = False
                self.perm = None
                print("NPY file loaded with default metadata (no sidecar .json found)")
                print("  Tip: Create a .json file with the same name to specify spacing/origin/direction")

        elif image_path.endswith('.nrrd') or image_path.endswith('.nhdr'):
            # Load NRRD file
            self.image_data, header = nrrd.read(image_path)

            # Get spacing and direction information
            try:
                space_directions = header.get('space directions', None)
                if space_directions is not None:
                    space_directions = np.array(space_directions)
                    # Calculate spacing as the magnitude (norm) of each direction vector
                    self.spacing = tuple(
                        np.linalg.norm(space_directions, axis=1))
                    # Direction matrix is the normalized space directions
                    spacing_array = np.array(self.spacing)
                    self.direction = space_directions / \
                        spacing_array[:, np.newaxis]
                elif 'spacings' in header:
                    # Fallback: use absolute values if spacings are provided directly
                    spacings = header['spacings']
                    self.spacing = tuple(np.abs(spacings))
                    self.direction = np.eye(3)
                else:
                    self.spacing = (1.0, 1.0, 1.0)
                    self.direction = np.eye(3)
            except Exception as e:
                print(f"Warning: Could not extract spacing from NRRD: {e}")
                self.spacing = (1.0, 1.0, 1.0)
                self.direction = np.eye(3)

            # Get origin
            try:
                self.origin = tuple(header.get(
                    'space origin', (0.0, 0.0, 0.0)))
            except:
                self.origin = (0.0, 0.0, 0.0)

            # Handle orientation
            try:
                # Create affine matrix from NRRD header
                if space_directions is not None:
                    affine = np.eye(4)
                    affine[:3, :3] = space_directions
                    affine[:3, 3] = self.origin

                    # Get orientation codes from affine using nibabel
                    orientation = nib.aff2axcodes(affine)
                    print(f"NRRD image orientation: {orientation}")
                    self.original_orientation = orientation

                    self.needs_sagittal_rotation = self._check_sagittal_rotation_needed(
                        orientation)
                    self.needs_coronal_rotation = self._check_coronal_rotation_needed(
                        orientation)

                    print(
                        f"Sagittal rotation needed: {self.needs_sagittal_rotation}")
                    print(
                        f"Coronal rotation needed: {self.needs_coronal_rotation}")
                else:
                    self.original_orientation = None
                    self.needs_sagittal_rotation = False
                    self.needs_coronal_rotation = False
            except Exception as e:
                print(f"Warning: Could not determine NRRD orientation: {e}")
                self.original_orientation = None
                self.needs_sagittal_rotation = False
                self.needs_coronal_rotation = False

            # Debug output
            print(
                f"NRRD dimensions: {self.image_data.shape}, spacing: {self.spacing}, origin: {self.origin}")
        elif image_path.lower().endswith('dicomdir'):
            # Handle DICOMDIR file
            try:
                # Read DICOMDIR
                dcm_dir = pydicom.dcmread(image_path)
                base_dir = os.path.dirname(image_path)

                # Method 1: Try hierarchical traversal
                series_dict = {}
                current_series = None

                for record in dcm_dir.DirectoryRecordSequence:
                    record_type = getattr(record, 'DirectoryRecordType', None)

                    if record_type == "SERIES":
                        # Get series information
                        series_uid = getattr(record, 'SeriesInstanceUID', None)
                        if not series_uid:
                            # Generate a unique ID if missing
                            series_uid = f"series_{len(series_dict)}"

                        current_series = series_uid
                        series_desc = getattr(
                            record, 'SeriesDescription', 'No Description')
                        series_number = getattr(
                            record, 'SeriesNumber', len(series_dict))

                        series_dict[series_uid] = {
                            'description': series_desc,
                            'number': series_number,
                            'images': []
                        }

                    elif record_type == "IMAGE":
                        # Try to get the referenced file path
                        try:
                            file_id = getattr(record, 'ReferencedFileID', None)
                            if file_id:
                                # Convert DICOM path format to OS path
                                if isinstance(file_id, (list, tuple)):
                                    file_path = os.path.join(
                                        base_dir, *file_id)
                                else:
                                    file_path = os.path.join(
                                        base_dir, str(file_id))

                                # Add to current series if we have one
                                if current_series and current_series in series_dict:
                                    if os.path.exists(file_path):
                                        series_dict[current_series]['images'].append(
                                            file_path)
                        except Exception as e:
                            print(
                                f"Warning: Could not process image record: {e}")
                            continue

                # Remove series with no images
                series_dict = {uid: info for uid,
                               info in series_dict.items() if info['images']}

                # Method 2: If no series found, try to scan the directory
                if not series_dict:
                    print("No series found via DICOMDIR parsing, scanning directory...")
                    # Recursively find all DICOM files
                    all_dicom_files = []
                    for root, dirs, files in os.walk(base_dir):
                        for file in files:
                            if not file.lower() == 'dicomdir':
                                file_path = os.path.join(root, file)
                                try:
                                    # Try to read as DICOM
                                    ds = pydicom.dcmread(
                                        file_path, stop_before_pixels=True)
                                    series_uid = getattr(
                                        ds, 'SeriesInstanceUID', None)
                                    if series_uid:
                                        if series_uid not in series_dict:
                                            series_dict[series_uid] = {
                                                'description': getattr(ds, 'SeriesDescription', 'No Description'),
                                                'number': getattr(ds, 'SeriesNumber', 0),
                                                'images': []
                                            }
                                        series_dict[series_uid]['images'].append(
                                            file_path)
                                except:
                                    continue

                # Show selection dialog
                if not series_dict:
                    raise ValueError(
                        "No DICOM series found in DICOMDIR or directory")

                selected_uid = DicomSeriesSelectionDialog.select_series(
                    series_dict)

                if selected_uid is None:
                    raise ValueError("No series selected")

                # Load the selected series
                dicom_files = series_dict[selected_uid]['images']
                if not dicom_files:
                    raise ValueError("Selected series has no images")

                self._load_dicom_series(dicom_files)

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise ValueError(f"Error loading DICOMDIR: {str(e)}")

        elif image_path.endswith('.dcm'):
            try:
                # Single DICOM file - treat as a single slice series
                ds = pydicom.dcmread(image_path)
                self.image_data = self._get_dicom_pixel_data(ds)
                
                # Add a third dimension for single slice (H, W) -> (H, W, 1)
                if self.image_data.ndim == 2:
                    self.image_data = self.image_data[:, :, np.newaxis]

                # Get spacing information
                try:
                    # [row_spacing, col_spacing]
                    pixel_spacing = ds.PixelSpacing
                    slice_thickness = getattr(ds, 'SliceThickness', 1.0)
                    self.spacing = (float(pixel_spacing[1]), float(
                        pixel_spacing[0]), float(slice_thickness))
                except:
                    self.spacing = (1.0, 1.0, 1.0)

                # Handle orientation using Image Orientation Patient (same as series)
                try:
                    iop = ds.ImageOrientationPatient
                    ipp = ds.ImagePositionPatient

                    # Create affine matrix from DICOM orientation
                    F = np.array(iop[:3])  # First row direction cosines
                    C = np.array(iop[3:])  # First column direction cosines

                    # Calculate normal vector (slice direction)
                    N = np.cross(F, C)

                    # Build affine transformation matrix
                    affine = np.eye(4)
                    affine[:3, 0] = F * self.spacing[0]
                    affine[:3, 1] = C * self.spacing[1]
                    affine[:3, 2] = N * self.spacing[2]
                    affine[:3, 3] = ipp

                    self.origin = affine[:3, 3]
                    self.direction = affine[:3, :3] / np.array(self.spacing)

                    # Get orientation
                    orientation = nib.aff2axcodes(affine)
                    print(f"Single DICOM orientation: {orientation}")
                    self.original_orientation = orientation

                    # Determine if rotation is needed for display
                    self.needs_sagittal_rotation = self._check_sagittal_rotation_needed(
                        orientation)
                    self.needs_coronal_rotation = self._check_coronal_rotation_needed(
                        orientation)
                    
                    print(f"Sagittal rotation needed: {self.needs_sagittal_rotation}")
                    print(f"Coronal rotation needed: {self.needs_coronal_rotation}")
                    
                    # For single slice, we typically don't transpose
                    # But we store the orientation for potential future use
                    self.perm = None
                    
                except AttributeError as e:
                    # If ImageOrientationPatient or ImagePositionPatient not available
                    print(f"Warning: Could not extract DICOM orientation metadata: {e}")
                    print("  Using default orientation (identity)")
                    self.original_orientation = None
                    self.needs_sagittal_rotation = False
                    self.needs_coronal_rotation = False
                    self.perm = None
                    self.origin = (0.0, 0.0, 0.0)
                    self.direction = np.eye(3)

            except Exception as e:
                raise ValueError(f"Error loading DICOM file: {str(e)}")

        elif os.path.isdir(image_path):
            try:
                # Directory containing multiple DICOM files (series)
                dicom_files = []
                for file in os.listdir(image_path):
                    file_path = os.path.join(image_path, file)
                    if os.path.isfile(file_path):
                        try:
                            # Try to read as DICOM
                            pydicom.dcmread(file_path)
                            dicom_files.append(file_path)
                        except:
                            # Not a valid DICOM file, skip
                            continue

                if not dicom_files:
                    raise ValueError(
                        "No DICOM files found in the specified directory")

                # Load the series with orientation handling
                self._load_dicom_series(dicom_files)

            except Exception as e:
                raise ValueError(f"Error loading DICOM directory: {str(e)}")
        else:
            raise ValueError(
                "Unsupported image format. Please use .dcm, .nii, .nii.gz, .npy files, or DICOMDIR.")

        self.pre_segmentation = None
        self.GT_segmentation = None
        shape = self.image_data.shape
        self.num_of_class = None
        if len(shape) == 3:
            self.hight, self.width, self.slices = shape
            self.processed_image = self.image_data.copy() if self.ww == -1 and self.wl == -1 else self.win_scale(
                self.image_data, self.wl, self.ww, np.uint8, [0, 255])
            return None
        elif len(shape) == 4:
            self.hight, self.width, self.slices = shape[:-1]
            self.image_data = np.squeeze(self.image_data[:, :, :, 0])
            self.processed_image = self.image_data.copy() if self.ww == -1 and self.wl == -1 else self.win_scale(
                self.image_data, self.wl, self.ww, np.uint8, [0, 255])
            return None
        else:
            self.image_data = None
            return shape

    def _load_dicom_series(self, dicom_files):
        """
        Helper method to load a DICOM series with proper orientation handling.
        :param dicom_files: List of DICOM file paths
        """
        # Sort files by instance number or slice location
        dicom_data = []
        for file_path in dicom_files:
            ds = pydicom.dcmread(file_path)
            slice_location = getattr(
                ds, 'SliceLocation', getattr(ds, 'InstanceNumber', 0))
            dicom_data.append((slice_location, ds))

        # Sort by slice location/instance number
        dicom_data.sort(key=lambda x: x[0])

        # Stack the pixel arrays
        pixel_arrays = [
            self._get_dicom_pixel_data(ds)
            for _, ds in dicom_data
        ]
        self.image_data = np.stack(pixel_arrays, axis=-1)

        # Get spacing and orientation from first slice
        first_ds = dicom_data[0][1]

        try:
            pixel_spacing = first_ds.PixelSpacing
            slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
            self.spacing = (float(pixel_spacing[1]), float(
                pixel_spacing[0]), float(slice_thickness))
        except:
            self.spacing = (1.0, 1.0, 1.0)

        # Handle orientation using Image Orientation Patient
        try:
            iop = first_ds.ImageOrientationPatient
            ipp = first_ds.ImagePositionPatient

            # Create affine matrix from DICOM orientation
            F = np.array(iop[:3])  # First row direction cosines
            C = np.array(iop[3:])  # First column direction cosines

            # Calculate normal vector (slice direction)
            N = np.cross(F, C)

            # Build affine transformation matrix
            affine = np.eye(4)
            affine[:3, 0] = F * self.spacing[0]
            affine[:3, 1] = C * self.spacing[1]
            affine[:3, 2] = N * self.spacing[2]
            affine[:3, 3] = ipp

            self.origin = affine[:3, 3]
            self.direction = affine[:3, :3] / np.array(self.spacing)

            # Get orientation and reorient if needed
            orientation = nib.aff2axcodes(affine)
            print(f"DICOM series orientation: {orientation}")
            self.original_orientation = orientation

            # Determine if rotation is needed for display BEFORE any transposition
            self.needs_sagittal_rotation = self._check_sagittal_rotation_needed(
                orientation)
            self.needs_coronal_rotation = self._check_coronal_rotation_needed(
                orientation)

            if orientation != ('S', 'C', 'A'):
                # Find current axis positions
                sag_axis = next(i for i, code in enumerate(
                    orientation) if code in ['L', 'R'])
                cor_axis = next(i for i, code in enumerate(
                    orientation) if code in ['A', 'P'])
                ax_axis = next(i for i, code in enumerate(
                    orientation) if code in ['S', 'I'])
                perm = (sag_axis, cor_axis, ax_axis)
                self.perm = perm
                self.image_data = np.transpose(self.image_data, perm)
                self.direction = self.direction[:, perm]
                self.spacing = tuple(self.spacing[i] for i in perm)
            else:
                self.perm = None
        except:
            # If orientation handling fails, keep original orientation
            self.original_orientation = None
            self.needs_sagittal_rotation = False
            self.needs_coronal_rotation = False
            self.perm = None
            self.origin = (0.0, 0.0, 0.0)
            self.direction = np.eye(3)
            pass

    def load_seg(self, image_path):
        """
        Load an image from a file.
        Check if segmentation match image size, return True 
        Otherwise return False
        :param image_path: Path to the image file.
        """
        # Load the image using nibabel
        if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
            self.GT_segmentation = nib.load(
                image_path).get_fdata().astype("int")
            # Only transpose if we have a valid permutation (when orientation was changed)
            if self.perm is not None:
                self.GT_segmentation = np.transpose(
                    self.GT_segmentation, self.perm)

        elif image_path.endswith('.nrrd') or image_path.endswith('.nhdr'):
            self.GT_segmentation, _ = nrrd.read(image_path)
            self.GT_segmentation = self.GT_segmentation.astype("int")
            # Only transpose if we have a valid permutation
            if self.perm is not None:
                self.GT_segmentation = np.transpose(
                    self.GT_segmentation, self.perm)

        elif image_path.endswith('.npy'):
            self.GT_segmentation = np.load(image_path).astype("int")

        elif image_path.endswith('.dcm'):
            # Handle DICOM files here if needed
            pass
        else:
            raise ValueError(
                "Unsupported image format. Please use .dcm, .nii, .nii.gz, or .npy files.")
        try:
            shape = self.image_data.shape
        except:
            return False

        values = np.unique(self.GT_segmentation)
        replacement_values = np.arange(len(values))
        value_map = dict(zip(values, replacement_values))
        vectorized_map = np.vectorize(value_map.get)
        self.GT_segmentation = vectorized_map(self.GT_segmentation)

        if len(self.GT_segmentation.shape) == 3:
            if self.GT_segmentation.shape == shape:
                self.num_of_class = len(np.unique(self.GT_segmentation))
                return True

        elif len(self.GT_segmentation.shape) == 4:
            if shape == self.GT_segmentation.shape[:-1]:
                self.GT_segmentation = np.squeeze(self.GT_segmentation[:, :, :, 0])
                self.num_of_class = len(np.unique(self.GT_segmentation))
                return True
        print(f"Segmentation shape {self.GT_segmentation.shape} does not match image shape {shape}.")
        self.GT_segmentation = None
        return False

    def get_image_shape(self):
        """
        Get the shape of the loaded image.
        :return: Shape of the image.
        """
        if self.image_data is not None:
            return self.image_data.shape
        else:
            raise ValueError("No image loaded. Please load an image first.")

    def get_image_spacing(self):
        """
        Get the spacing of the loaded image.
        :return: Spacing of the image.
        """
        if self.spacing is not None:
            return self.spacing
        else:
            raise ValueError("No image loaded. Please load an image first.")

    def get_origin(self):
        """
        Get the origin of the loaded image.
        :return: Origin of the image.
        """
        if self.origin is not None:
            return self.origin
        else:
            raise ValueError("No image loaded. Please load an image first.")

    def get_direction(self):
        """
        Get the direction of the loaded image.
        :return: Direction of the image.
        """
        if self.direction is not None:
            return self.direction
        else:
            raise ValueError("No image loaded. Please load an image first.")

    def resize_image(self, new_shape, netwoerkType, mode):
        """
        Resize the image to a new shape.
        :param new_shape: New shape for the image.
        """
        proceessed_image = self.processed_image.copy()
        append = ""
        if "-" in mode:
            append = mode.split("-")[0]
            mode = mode.split("-")[1]

        if mode == "append_last":
            proceessed_image = np.expand_dims(proceessed_image, axis=-1)

        elif mode == "append_first":
            proceessed_image = np.expand_dims(proceessed_image, axis=0)

        elif mode == "padd":
            if self.image_data is not None:
                pad_h = new_shape[0] - self.hight
                pad_w = new_shape[1] - self.width
                pad_s = new_shape[2] - \
                    self.slices if netwoerkType == "3D" else 0
                proceessed_image = np.pad(proceessed_image, ((
                    0, pad_h), (0, pad_w), (pad_s, 0)), mode='constant', constant_values=0)
        elif mode == "resize":
            if self.image_data is not None:
                if netwoerkType == "2D":
                    proceessed_image = self.resize_volume_slices(
                        proceessed_image, (new_shape[0], new_shape[1]))
                elif netwoerkType == "3D":
                    proceessed_image = resize(
                        proceessed_image, new_shape, preserve_range=True, anti_aliasing=True)

        elif mode == "rotate":
            if self.image_data is not None:
                ndim = len(proceessed_image.shape)
                if ndim == 3:
                    return np.transpose(proceessed_image, (2, 0, 1))
                elif ndim == 4:
                    return np.transpose(proceessed_image, (3, 0, 1, 2))

        if append:
            proceessed_image = np.expand_dims(
                proceessed_image, axis=-1) if append == "append_last" else np.expand_dims(self.image_data, axis=0)
        return proceessed_image

    @staticmethod
    def fit_to_model_shape(image_data, required_shape, crop_positions=None):
        """Fit spatial dimensions by cropping or zero-padding."""
        image_data = np.asarray(image_data)
        required_shape = tuple(required_shape)
        crop_positions = crop_positions or {}

        if image_data.ndim != 3 or len(required_shape) not in (2, 3):
            raise ValueError("Expected a 3D volume and a 2D or 3D spatial shape")

        slices = []
        padding = []
        axes = []
        for axis, current in enumerate(image_data.shape):
            target = required_shape[axis] if axis < len(required_shape) else current
            target = current if target is None else int(target)
            if target <= 0:
                raise ValueError(f"Invalid size on axis {axis}: {target}")

            crop_start = 0
            pad_before = 0
            pad_after = 0
            if current > target:
                position = crop_positions.get(axis, "center")
                difference = current - target
                if position == "start":
                    crop_start = 0
                elif position == "center":
                    crop_start = difference // 2
                elif position == "end":
                    crop_start = difference
                else:
                    raise ValueError(f"Invalid crop position: {position}")
            elif current < target:
                difference = target - current
                pad_before = difference // 2
                pad_after = difference - pad_before

            crop_end = crop_start + min(current, target)
            slices.append(slice(crop_start, crop_end))
            padding.append((pad_before, pad_after))
            axes.append({
                "input_length": current,
                "crop_start": crop_start,
                "crop_end": crop_end,
                "pad_before": pad_before,
                "pad_after": pad_after,
            })

        fitted = np.pad(
            image_data[tuple(slices)],
            padding,
            mode="constant",
            constant_values=0,
        )
        transform = {
            "original_shape": tuple(image_data.shape),
            "fitted_shape": tuple(fitted.shape),
            "axes": axes,
        }
        return fitted, transform

    @staticmethod
    def restore_from_model_shape(segmentation, transform):
        """Undo the recorded crop and padding without stretching labels."""
        segmentation = np.asarray(segmentation)
        if tuple(segmentation.shape) != transform["fitted_shape"]:
            raise ValueError(
                f"Prediction shape {segmentation.shape} does not match "
                f"model input shape {transform['fitted_shape']}"
            )

        unpad_slices = []
        destination_slices = []
        for axis in transform["axes"]:
            start = axis["pad_before"]
            end = segmentation.shape[len(unpad_slices)] - axis["pad_after"]
            unpad_slices.append(slice(start, end))
            destination_slices.append(slice(axis["crop_start"], axis["crop_end"]))

        restored = np.zeros(transform["original_shape"], dtype=segmentation.dtype)
        restored[tuple(destination_slices)] = segmentation[tuple(unpad_slices)]
        return restored

    @staticmethod
    def restore_resampled_segmentation(segmentation, original_shape):
        """Map labels back to the original voxel grid using nearest neighbors."""
        segmentation = np.asarray(segmentation)
        original_shape = tuple(original_shape)
        if tuple(segmentation.shape) == original_shape:
            return segmentation
        if segmentation.ndim != 3 or len(original_shape) != 3:
            raise ValueError("Expected 3D segmentation and target shape")
        zoom_factors = [target / current for current, target in
                        zip(segmentation.shape, original_shape)]
        restored = ndimage.zoom(segmentation, zoom_factors, order=0, mode="nearest")
        if tuple(restored.shape) != original_shape:
            raise ValueError(f"Resampling produced unexpected shape: {restored.shape}")
        return restored.astype(segmentation.dtype, copy=False)

    def crop_image(self, start1, end1, start2, end2, start3, end3):
        """
        Crop the image to a specified range.
        :param start: Start index for cropping.
        :param end: End index for cropping.
        """
        if self.image_data is not None:
            self.image_data = self.image_data[start1:end1,
                                              start2:end2, start3:end3]
            self.hight, self.width, self.slices = self.image_data.shape
        else:
            raise ValueError("No image loaded. Please load an image first.")

    def win_scale(self, data, wl, ww, dtype=np.uint8, out_range=(0, 255)):
        """
        Apply DICOM-style linear windowing.

        Parameters
        ----------
        data : np.ndarray
            Image in modality units, such as HU for CT.
        wl : float
            Window level/center.
        ww : float
            Window width.
        dtype : np.dtype
            Output datatype.
        out_range : tuple
            Minimum and maximum output intensities.
        """
        wl = float(wl)
        ww = float(ww)

        if ww < 1:
            raise ValueError("Window width must be at least 1.")

        data = np.asarray(data, dtype=np.float32)
        out_min, out_max = map(float, out_range)

        lower = wl - 0.5 - (ww - 1.0) / 2.0
        upper = wl - 0.5 + (ww - 1.0) / 2.0

        if ww == 1:
            scaled = np.where(data <= lower, out_min, out_max)
        else:
            scaled = np.empty_like(data, dtype=np.float32)

            below = data <= lower
            above = data > upper
            middle = ~(below | above)

            scaled[below] = out_min
            scaled[above] = out_max

            scaled[middle] = (
                ((data[middle] - (wl - 0.5)) / (ww - 1.0) + 0.5)
                * (out_max - out_min)
                + out_min
            )

        scaled = np.clip(scaled, out_min, out_max)
        return scaled.astype(dtype)
    def get_processed_image(self):
        """
        Get the processed image.
        :return: Processed image.
        """
        if self.processed_image is not None:
            return self.processed_image

    def get_image_data(self):
        """
        Get the original image data.
        :return: Original image data.
        """
        if self.image_data is not None:
            return self.image_data

    def get_GT_segmentation(self):
        """
        Get the segmentation of the image.
        :return: Segmentation of the image.
        """
        if self.GT_segmentation is not None:
            return self.GT_segmentation

    def get_pre_segmentation(self):
        """
        Get the segmentation of the image.
        :return: Segmentation of the image.
        """
        if self.pre_segmentation is not None:
            return self.pre_segmentation

    def set_window_level(self, wl, ww):
        self.ww = ww
        self.wl = wl
        if ww == -1 and wl == -1:
            self.processed_image = self.image_data.copy()
        else:
            self.processed_image = self.win_scale(
                self.image_data, wl, ww, np.uint8, [0, 255])

    def get_num_of_classes(self):
        return self.num_of_class

    def _check_sagittal_rotation_needed(self, orientation):
        """
        Check if sagittal view needs rotation for proper medical display.
        Sagittal slices along first axis, displaying axes 1 and 2.
        Standard medical display: Anterior at top, Superior at right
        """
        # For sagittal view, we're looking at axes 1 and 2
        # Typically want: vertical axis pointing A->P, horizontal axis pointing I->S
        # Common orientations that need rotation:
        # - If orientation[1] is 'A'/'P' and orientation[2] is 'I'/'S': likely needs rotation
        # - Standard RAS or LAS: typically needs rotation

        # Simple heuristic: if it's standard radiological orientation, usually needs rotation
        if orientation in [('R', 'A', 'S'), ('L', 'A', 'S'), ('R', 'P', 'S'), ('L', 'P', 'S')]:
            return True
        return False

    def _check_coronal_rotation_needed(self, orientation):
        """
        Check if coronal view needs rotation for proper medical display.
        Coronal slices along second axis, displaying axes 0 and 2.
        Standard medical display: Left at left, Superior at top
        """
        # For coronal view, we're looking at axes 0 and 2
        # Typically want: vertical axis pointing I->S, horizontal axis pointing R->L

        # Simple heuristic: if it's standard radiological orientation, usually needs rotation
        if orientation in [('R', 'A', 'S'), ('L', 'A', 'S'), ('R', 'P', 'S'), ('L', 'P', 'S')]:
            return True
        return False

    def get_sagittal_rotation_needed(self):
        """Get whether sagittal view needs rotation"""
        return getattr(self, 'needs_sagittal_rotation', False)

    def get_coronal_rotation_needed(self):
        """Get whether coronal view needs rotation"""
        return getattr(self, 'needs_coronal_rotation', False)

    def set_pre_segmentation(self, pre_seg, netwoerkType, strict=False):
        """Store a segmentation in the original image voxel grid."""
        pre_seg = np.asarray(pre_seg)
        target_shape = (self.hight, self.width, self.slices)
        if pre_seg.ndim != 3:
            raise ValueError(f"Expected a 3D segmentation, got {pre_seg.shape}")

        if tuple(pre_seg.shape) == target_shape:
            resized_mask = pre_seg
        elif strict:
            raise ValueError(
                f"Segmentation shape {pre_seg.shape} does not match image {target_shape}"
            )
        elif netwoerkType == "2D":
            resized_mask = self.resize_volume_slices(pre_seg, (self.width, self.hight))
        elif netwoerkType == "3D":
            resized_mask = resize(
                pre_seg, target_shape, order=0,
                preserve_range=True, anti_aliasing=False
            ).astype(pre_seg.dtype)
        else:
            raise ValueError(f"Unknown network dimensionality: {netwoerkType}")

        self.pre_segmentation = resized_mask
        self.num_of_class = len(np.unique(resized_mask))

    def resize_volume_slices(self, volume, new_size):
        """
        Resize each slice of a 3D volume (H, W, D) → (new_H, new_W, D)
        using nearest neighbor (safe for masks).
        """
        h, w, d = volume.shape
        resized_slices = []
        for i in range(d):
            slice_2d = volume[:, :, i]
            img = PILImage.fromarray(slice_2d.astype(np.uint8))
            img_resized = img.resize(new_size, resample=PILImage.NEAREST)
            resized_slices.append(np.array(img_resized))
        return np.stack(resized_slices, axis=-1)

    def restore_size(self, volume, mode, netwoerkType):
        """
        Restore the original size of a 3D volume (H, W, D) → (orig_H, orig_W, orig_D)
        """
        original_shape = (self.hight, self.width, self.slices)
        restored_volume = volume.squeeze()
        if mode == "padd":
            restored_volume = volume[0:original_shape[0],
                                     0:original_shape[1], 0:original_shape[2]]
        elif mode == "resize":
            if netwoerkType == "2D":
                restored_volume = self.resize_volume_slices(
                    volume, (original_shape[1], original_shape[0]))
            elif netwoerkType == "3D":
                restored_volume = resize(
                    volume, original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(volume.dtype)
        elif mode == "rotate":
            ndim = len(volume.shape)
            if ndim == 3:
                restored_volume = np.transpose(volume, (1, 2, 0))
            elif ndim == 4:
                restored_volume = np.transpose(volume, (1, 2, 3, 0))
        return restored_volume

    def _nrrd_space_to_orientation(self, space):
        """
        Convert NRRD space descriptor to orientation codes.
        :param space: NRRD space string (e.g., 'left-posterior-superior')
        :return: Tuple of orientation codes
        """
        space_map = {
            'left-posterior-superior': ('L', 'P', 'S'),
            'right-anterior-superior': ('R', 'A', 'S'),
            'left-anterior-superior': ('L', 'A', 'S'),
            'right-posterior-superior': ('R', 'P', 'S'),
        }

        return space_map.get(space.lower(), ('R', 'A', 'S'))  # Default to RAS
    
    @staticmethod
    def _get_dicom_pixel_data(ds):
        """
        Decode DICOM pixels and convert stored values to modality values,
        such as Hounsfield units for CT.
        """
        pixels = ds.pixel_array
        pixels = apply_modality_lut(pixels, ds)
        return np.asarray(pixels, dtype=np.float32)