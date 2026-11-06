import numpy as np
from typing import Dict, Any, Optional, Union
import logging
from scipy import ndimage
from scipy.ndimage import zoom

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingExecutor:
    """
    Class to execute preprocessing functions based on the dialog output.
    """

    def __init__(self):
        """Initialize the preprocessing executor."""
        # Map option names to their corresponding functions
        self.function_mapping = {
            "Z-Normalization (Based on This Image)": self._apply_z_normalization_image,
            "Z-Normalization (You need to enter avr/std)": self._apply_z_normalization_manual,
            "Min/Max Normalization (Enter range)": self._apply_minmax_normalization,
            "Gaussian Blur": self._apply_gaussian_blur,
            "Isotropic Resampling": self._apply_isotropic_resampling
        }

    def execute_preprocessing(self, image_data: np.ndarray, options_dict: Dict[str, Any]) -> np.ndarray:
        """
        Execute preprocessing functions based on the options dictionary from the dialog.

        Args:
            image_data (np.ndarray): Input image data
            options_dict (Dict[str, Any]): Dictionary from preprocessing dialog

        Returns:
            np.ndarray: Processed image data
        """
        processed_image = image_data.copy()

        logger.info("Starting preprocessing pipeline...")
        logger.info(f"Input image shape: {processed_image.shape}")

        # Spatial resampling must precede intensity preprocessing.
        ordered_options = list(options_dict.items())
        ordered_options.sort(key=lambda item: 0 if item[0] == "Isotropic Resampling" else 1)
        for option_name, option_data in ordered_options:
            if isinstance(option_data, dict) and option_data.get("enabled", False):
                if option_name in self.function_mapping:
                    try:
                        logger.info(f"Applying: {option_name}")

                        # Get the function for this option
                        preprocessing_func = self.function_mapping[option_name]

                        # Extract values if they exist
                        values = option_data.get("values", None)

                        # Call the function with appropriate parameters
                        processed_image = preprocessing_func(
                            processed_image, values)

                        logger.info(f"Successfully applied: {option_name}")
                        logger.info(f"Output image shape: {processed_image.shape}")

                    except Exception as e:
                        logger.error(f"Error applying {option_name}: {str(e)}")
                        raise
                else:
                    logger.warning(
                        f"No function found for option: {option_name}")

        logger.info("Preprocessing pipeline completed!")
        return processed_image

    def _apply_z_normalization_image(self, image: np.ndarray, values: Optional[list]) -> np.ndarray:
        """
        Apply Z-normalization based on the image's own statistics.
        """
        mean = np.mean(image)
        std = np.std(image)

        logger.info(f"Z-normalization (image-based): mean={mean:.4f}, std={std:.4f}")

        if std == 0:
            logger.warning("Standard deviation is 0, returning original image")
            return image

        normalized_image = (image - mean) / std
        return normalized_image

    def _apply_z_normalization_manual(self, image: np.ndarray, values: Optional[list]) -> np.ndarray:
        """
        Apply Z-normalization with manually provided mean and std.

        Args:
            image: Input image
            values: [mean, std]
        """
        if values is None or len(values) != 2:
            raise ValueError(
                "Manual Z-normalization requires exactly 2 values: [mean, std]")

        mean, std = values

        logger.info(f"Z-normalization (manual): mean={mean}, std={std}")

        if std == 0:
            raise ValueError("Standard deviation cannot be 0")

        normalized_image = (image - mean) / std
        return normalized_image

    def _apply_minmax_normalization(self, image: np.ndarray, values: Optional[list]) -> np.ndarray:
        """
        Apply Min/Max normalization to specified range.

        Args:
            image: Input image
            values: [min_value, max_value]
        """
        if values is None or len(values) != 2:
            raise ValueError(
                "Min/Max normalization requires exactly 2 values: [min_value, max_value]")

        target_min, target_max = values

        # Get current min and max
        current_min = np.min(image)
        current_max = np.max(image)

        logger.info(f"Min/Max normalization: [{current_min:.4f}, {current_max:.4f}] -> [{target_min}, {target_max}]")

        if current_max == current_min:
            logger.warning(
                "Image has constant values, returning image filled with target_min")
            return np.full_like(image, target_min)

        # Normalize to target range
        normalized_image = (image - current_min) / (current_max - current_min)
        normalized_image = normalized_image * \
            (target_max - target_min) + target_min

        return normalized_image

    def _apply_gaussian_blur(self, image: np.ndarray, values: Optional[list]) -> np.ndarray:
        """
        Apply Gaussian blur to the image.

        Args:
            image: Input image
            values: [sigma] or [sigma_x, sigma_y, sigma_z]
        """
        if values is None or len(values) == 0:
            raise ValueError(
                "Gaussian blur requires at least 1 value: [sigma]")

        if len(values) == 1:
            sigma = values[0]
            logger.info(f"Applying Gaussian blur with sigma={sigma}")
        else:
            sigma = values  # Multiple sigma values for different axes
            logger.info(f"Applying Gaussian blur with sigma={sigma}")

        blurred_image = ndimage.gaussian_filter(image, sigma=sigma)
        return blurred_image

    def _apply_isotropic_resampling(self, image: np.ndarray, values: Optional[list]) -> np.ndarray:
        """
        Apply isotropic resampling to make voxel spacing uniform.

        Args:
            image: Input image
            values: [target_spacing] or [current_spacing_x, current_spacing_y, current_spacing_z, target_spacing]
        
        Examples:
            - values=[1.0]: Assumes current spacing is [1,1,1], resample to 1mm isotropic
            - values=[0.5, 0.5, 2.0, 1.0]: Current spacing is [0.5,0.5,2.0]mm, resample to 1mm isotropic
        """
        if values is None or len(values) == 0:
            raise ValueError(
                "Isotropic resampling requires spacing values. "
                "Provide [target_spacing] or [current_x, current_y, current_z, target_spacing]")

        if len(values) == 1:
            # Assume current spacing is [1, 1, 1] and target is values[0]
            target_spacing = values[0]
            current_spacing = [1.0, 1.0, 1.0]
            logger.info(f"Isotropic resampling: assuming current spacing [1,1,1]mm -> target {target_spacing}mm")
        elif len(values) == 4:
            # [current_x, current_y, current_z, target]
            current_spacing = values[:3]
            target_spacing = values[3]
            logger.info(f"Isotropic resampling: {current_spacing}mm -> {target_spacing}mm isotropic")
        else:
            raise ValueError(
                "Invalid number of values for isotropic resampling. "
                "Provide [target_spacing] or [current_x, current_y, current_z, target_spacing]")

        if image.ndim != 3:
            raise ValueError(f"Isotropic resampling requires a 3D image: {image.shape}")
        current_spacing = np.asarray(current_spacing, dtype=float)
        target_spacing = float(target_spacing)
        if not np.all(np.isfinite(current_spacing)) or np.any(current_spacing <= 0):
            raise ValueError("Original voxel spacing must contain positive finite values")
        if not np.isfinite(target_spacing) or target_spacing <= 0:
            raise ValueError("Target voxel spacing must be positive and finite")

        # Keep physical extents approximately unchanged.
        zoom_factors = (current_spacing / target_spacing).tolist()
        
        logger.info(f"Zoom factors: {zoom_factors}")
        logger.info(f"Original shape: {image.shape}")

        # Perform resampling with cubic interpolation (order=3)
        try:
            resampled_image = zoom(image, zoom_factors, order=3, mode='nearest')
            logger.info(f"Resampled shape: {resampled_image.shape}")
        except Exception as e:
            logger.error(f"Resampling failed: {str(e)}")
            logger.info("Falling back to linear interpolation (order=1)")
            resampled_image = zoom(image, zoom_factors, order=1, mode='nearest')

        return resampled_image


def execute_preprocessing_pipeline(image_data: np.ndarray, preprocessing_options: Dict[str, Any]) -> np.ndarray:
    """
    Convenience function to execute the preprocessing pipeline.

    Args:
        image_data (np.ndarray): Input image data
        preprocessing_options (Dict[str, Any]): Options dictionary from the dialog

    Returns:
        np.ndarray: Processed image data
    """
    executor = PreprocessingExecutor()
    return executor.execute_preprocessing(image_data, preprocessing_options)
