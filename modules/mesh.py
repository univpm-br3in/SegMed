import numpy as np
from skimage import measure
from stl import mesh
import os
import logging

logger = logging.getLogger(__name__)


class Mesh:
    def __init__(self):
        self.spacing = None
        self.verticesGT = None
        self.facesGT = None
        self.classesGT = None
        self.verticesPre = None
        self.facesPre = None
        self.classesPre = None
        self.pre_mesh_items = None
        self.gt_mesh_items = None
        self.origin = None
        self.direction = None

    def set_spacing(self, spacing):
        """
        Set the spacing for the mesh.
        :param spacing: spacing in millimeters (tuple or list of 3 values).
        """
        if spacing is None:
            logger.warning("Setting spacing to default [1.0, 1.0, 1.0]")
            self.spacing = [1.0, 1.0, 1.0]
        else:
            self.spacing = spacing
            logger.info(f"Spacing set to: {self.spacing}")

    def set_origin(self, origin):
        """
        Set the physical origin for the mesh.
        :param origin: origin coordinates (tuple or list of 3 values).
        """
        if origin is None:
            logger.warning("Setting origin to default [0.0, 0.0, 0.0]")
            self.origin = [0.0, 0.0, 0.0]
        else:
            self.origin = origin
            logger.info(f"Origin set to: {self.origin}")

    def set_direction(self, direction):
        """
        Set the direction/orientation matrix for the mesh.
        :param direction: 3x3 direction matrix.
        """
        if direction is None:
            logger.warning("Setting direction to identity matrix")
            self.direction = np.eye(3)
        elif not isinstance(direction, np.ndarray):
            self.direction = np.array(direction)
        else:
            self.direction = direction
        
        # Validate direction matrix
        if self.direction.shape != (3, 3):
            logger.error(f"Invalid direction matrix shape: {self.direction.shape}, using identity")
            self.direction = np.eye(3)
        else:
            logger.info(f"Direction matrix set")

    def create_faces_triangles_for_classes(self, volume):
        """
        Create a combined mesh from a 3D volume for all classes, mapped to physical coordinates.

        :param volume: 3D numpy array representing the segmented volume.
        :return: vertices, faces, and a vector specifying the class of each face.
        """
        # Validate input
        if volume is None:
            logger.error("Volume is None, cannot create mesh")
            return None, None, None
        
        if volume.size == 0:
            logger.error("Volume is empty, cannot create mesh")
            return None, None, None
        
        if len(volume.shape) != 3:
            logger.error(f"Volume must be 3D, got shape {volume.shape}")
            return None, None, None

        # Extract geometry info with defaults
        spacing = self.spacing if self.spacing is not None else [1.0, 1.0, 1.0]
        origin = self.origin if self.origin is not None else [0.0, 0.0, 0.0]
        direction = self.direction if self.direction is not None else np.eye(3)

        # Validate and convert to numpy arrays
        spacing = np.array(spacing)
        origin = np.array(origin)
        
        if not isinstance(direction, np.ndarray):
            direction = np.array(direction)
        
        # Final validation
        if direction.shape != (3, 3):
            logger.warning(f"Invalid direction matrix shape: {direction.shape}, using identity")
            direction = np.eye(3)

        logger.info(f"Creating mesh with spacing={spacing}, origin={origin}")
        logger.info(f"Volume shape: {volume.shape}")

        unique_classes = np.unique(volume)
        logger.info(f"Found {len(unique_classes)} unique classes: {unique_classes}")

        all_vertices = []
        all_faces = []
        face_classes = []

        vertex_offset = 0

        for class_value in unique_classes:
            if class_value == 0:
                continue  # Skip background

            logger.info(f"Processing class {class_value}...")
            
            # Binary mask for this class
            binary = (volume == class_value).astype(np.uint8)
            
            # Check if there are any voxels for this class
            if np.sum(binary) == 0:
                logger.warning(f"Class {class_value} has no voxels, skipping")
                continue

            try:
                # Run marching cubes (voxel space scaled by spacing)
                vertices, faces, _, _ = measure.marching_cubes(
                    binary, level=0.5, spacing=tuple(spacing)
                )
                
                logger.info(f"Class {class_value}: generated {len(vertices)} vertices, {len(faces)} faces")

                # Map to physical space: apply direction + origin
                # vertices shape: (N, 3), direction shape: (3, 3)
                try:
                    vertices = (direction @ vertices.T).T + origin
                except ValueError as e:
                    logger.error(f"Error applying direction matrix: {e}")
                    logger.info("Skipping direction transformation")
                    vertices = vertices + origin

                # Offset faces to account for previous vertices
                faces = faces + vertex_offset

                # Append results
                all_vertices.append(vertices)
                all_faces.append(faces)
                face_classes.extend([class_value] * len(faces))

                vertex_offset += len(vertices)

            except Exception as e:
                logger.error(f"Error processing class {class_value}: {str(e)}")
                continue

        if all_vertices and all_faces:
            final_vertices = np.vstack(all_vertices)
            final_faces = np.vstack(all_faces)
            final_classes = np.array(face_classes)
            
            logger.info(f"Mesh creation complete: {len(final_vertices)} total vertices, "
                       f"{len(final_faces)} total faces")
            
            return final_vertices, final_faces, final_classes
        else:
            logger.warning("No mesh data generated")
            return None, None, None

    def create_faces_triangles_GT(self, volume):
        """Create mesh for ground truth segmentation."""
        logger.info("Creating ground truth mesh...")
        self.verticesGT, self.facesGT, self.classesGT = self.create_faces_triangles_for_classes(
            volume)
        
        if self.verticesGT is not None:
            logger.info(f"GT mesh created successfully")
        else:
            logger.warning("GT mesh creation failed")

    def create_faces_triangles_pre(self, volume):
        """Create mesh for predicted segmentation."""
        logger.info("Creating prediction mesh...")
        self.verticesPre, self.facesPre, self.classesPre = self.create_faces_triangles_for_classes(
            volume)
        
        if self.verticesPre is not None:
            logger.info(f"Prediction mesh created successfully")
        else:
            logger.warning("Prediction mesh creation failed")

    def export_stl(self, filename, gtFlag, enabled_classes=None):
        """
        Export the mesh to a .stl file.

        :param filename: Path to the output .stl file.
        :param gtFlag: True for ground truth, False for prediction.
        :param enabled_classes: List of boolean values indicating which classes to export (optional).
        """
        if gtFlag:
            vertices = self.verticesGT
            faces = self.facesGT
            classes = self.classesGT
            mesh_type = "ground truth"
        else:
            vertices = self.verticesPre
            faces = self.facesPre
            classes = self.classesPre
            mesh_type = "prediction"

        if vertices is None or faces is None:
            logger.error(f"Cannot export {mesh_type} STL: mesh data is None")
            raise ValueError(f"No {mesh_type} mesh data available")

        # Filter by enabled classes if provided
        if enabled_classes is not None and classes is not None:
            # Create mask for enabled classes (class indices start at 1)
            mask = np.zeros(len(classes), dtype=bool)
            unique_classes = np.unique(classes)
            
            for cls in unique_classes:
                class_idx = int(cls) - 1  # Convert to 0-based index
                if class_idx < len(enabled_classes) and enabled_classes[class_idx]:
                    mask |= (classes == cls)
            
            if not np.any(mask):
                logger.warning("No enabled classes to export")
                return
            
            # Filter faces by enabled classes
            faces = faces[mask]
            
            # Find unique vertices used by remaining faces
            unique_vertex_indices, inverse_indices = np.unique(faces, return_inverse=True)
            vertices = vertices[unique_vertex_indices]
            faces = inverse_indices.reshape(faces.shape)

        logger.info(f"Exporting {mesh_type} mesh to {filename}...")

        # Create a new mesh
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

        for i, f in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = vertices[f[j], :]

        # Write the mesh to file
        stl_mesh.save(filename)
        logger.info(f"STL file saved successfully: {filename}")

    def export_stl_files(self, output_dir, gtFlag, enabled_classes=None, class_names=None):
        """
        Export the mesh to separate STL files for each class.

        :param output_dir: Directory to save the STL files.
        :param gtFlag: True for ground truth, False for prediction.
        :param enabled_classes: List of boolean values indicating which classes to export (optional).
        :param class_names: List of custom names for each class (optional).
        """
        if gtFlag:
            vertices = self.verticesGT
            faces = self.facesGT
            classes = self.classesGT
            mesh_type = "ground truth"
        else:
            vertices = self.verticesPre
            faces = self.facesPre
            classes = self.classesPre
            mesh_type = "prediction"

        if vertices is None or faces is None or classes is None:
            logger.error(f"Cannot export {mesh_type} STL files: mesh data is None")
            raise ValueError(f"No {mesh_type} mesh data available")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Exporting {mesh_type} mesh by class to {output_dir}...")

        unique_classes = np.unique(classes)
        
        for cls in unique_classes:
            class_idx = int(cls) - 1  # Convert to 0-based index
            
            # Check if this class is enabled
            if enabled_classes is not None and class_idx < len(enabled_classes):
                if not enabled_classes[class_idx]:
                    logger.info(f"Skipping disabled class {cls}")
                    continue
            
            # Select triangles of the current class
            class_mask = classes == cls
            class_triangles = faces[class_mask]

            if len(class_triangles) == 0:
                logger.warning(f"Class {cls} has no faces, skipping")
                continue

            # Find unique vertex indices and remap
            unique_vertex_indices, inverse_indices = np.unique(
                class_triangles, return_inverse=True)
            filtered_vertices = vertices[unique_vertex_indices]
            remapped_triangles = inverse_indices.reshape(class_triangles.shape)

            # Create data array for STL mesh
            class_mesh = mesh.Mesh(
                np.zeros(len(remapped_triangles), dtype=mesh.Mesh.dtype))
            for i, face in enumerate(remapped_triangles):
                for j in range(3):
                    class_mesh.vectors[i][j] = filtered_vertices[face[j]]

            # Determine filename - use custom name if provided
            if class_names is not None and class_idx < len(class_names):
                filename = f"{class_names[class_idx]}.stl"
            else:
                filename = f"class_{cls}.stl"
            
            # Save STL
            file_path = os.path.join(output_dir, filename)
            class_mesh.save(file_path)
            logger.info(f"Exported class {cls} to {file_path}")

    def export_obj_files(self, output_dir, gtFlag, enabled_classes=None, class_names=None):
        """
        Export the mesh to separate OBJ files for each class.

        :param output_dir: Directory to save the OBJ files.
        :param gtFlag: True for ground truth, False for prediction.
        :param enabled_classes: List of boolean values indicating which classes to export (optional).
        :param class_names: List of custom names for each class (optional).
        """
        if gtFlag:
            vertices = self.verticesGT
            faces = self.facesGT
            classes = self.classesGT
            mesh_type = "ground truth"
        else:
            vertices = self.verticesPre
            faces = self.facesPre
            classes = self.classesPre
            mesh_type = "prediction"

        if vertices is None or faces is None or classes is None:
            logger.error(f"Cannot export {mesh_type} OBJ files: mesh data is None")
            raise ValueError(f"No {mesh_type} mesh data available")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Exporting {mesh_type} mesh by class to {output_dir}...")

        unique_classes = np.unique(classes)
        
        for cls in unique_classes:
            class_idx = int(cls) - 1  # Convert to 0-based index
            
            # Check if this class is enabled
            if enabled_classes is not None and class_idx < len(enabled_classes):
                if not enabled_classes[class_idx]:
                    logger.info(f"Skipping disabled class {cls}")
                    continue
            
            # Select triangles of the current class
            class_mask = classes == cls
            class_triangles = faces[class_mask]
            
            if len(class_triangles) == 0:
                logger.warning(f"Class {cls} has no faces, skipping")
                continue
            
            # Find unique vertex indices and remap
            unique_vertex_indices, inverse_indices = np.unique(
                class_triangles, return_inverse=True)
            filtered_vertices = vertices[unique_vertex_indices]
            remapped_triangles = inverse_indices.reshape(class_triangles.shape)
            
            # Create OBJ file content
            obj_content = []
            obj_content.append(f"# OBJ file for class {cls}")
            for vertex in filtered_vertices:
                obj_content.append("v {} {} {}".format(
                    vertex[0], vertex[1], vertex[2]))
            for face in remapped_triangles:
                obj_content.append("f {} {} {}".format(
                    face[0] + 1, face[1] + 1, face[2] + 1))
            
            # Determine filename - use custom name if provided
            if class_names is not None and class_idx < len(class_names):
                filename = f"{class_names[class_idx]}.obj"
            else:
                filename = f"class_{cls}.obj"
            
            # Save OBJ file
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w') as obj_file:
                obj_file.write("\n".join(obj_content))
            
            logger.info(f"Exported class {cls} to {file_path}")

    def export_obj(self, filename, gtFlag, enabled_classes=None):
        """
        Export the mesh to a .obj file.

        :param filename: Path to the output .obj file.
        :param gtFlag: True for ground truth, False for prediction.
        :param enabled_classes: List of boolean values indicating which classes to export (optional).
        """
        if gtFlag:
            vertices = self.verticesGT
            faces = self.facesGT
            classes = self.classesGT
            mesh_type = "ground truth"
        else:
            vertices = self.verticesPre
            faces = self.facesPre
            classes = self.classesPre
            mesh_type = "prediction"

        if vertices is None or faces is None:
            logger.error(f"Cannot export {mesh_type} OBJ: mesh data is None")
            raise ValueError(f"No {mesh_type} mesh data available")

        # Filter by enabled classes if provided
        if enabled_classes is not None and classes is not None:
            # Create mask for enabled classes (class indices start at 1)
            mask = np.zeros(len(classes), dtype=bool)
            unique_classes = np.unique(classes)
            
            for cls in unique_classes:
                class_idx = int(cls) - 1  # Convert to 0-based index
                if class_idx < len(enabled_classes) and enabled_classes[class_idx]:
                    mask |= (classes == cls)
            
            if not np.any(mask):
                logger.warning("No enabled classes to export")
                return
            
            # Filter faces by enabled classes
            faces = faces[mask]
            
            # Find unique vertices used by remaining faces
            unique_vertex_indices, inverse_indices = np.unique(faces, return_inverse=True)
            vertices = vertices[unique_vertex_indices]
            faces = inverse_indices.reshape(faces.shape)

        logger.info(f"Exporting {mesh_type} mesh to {filename}...")

        # Create OBJ file content
        obj_content = []
        obj_content.append("# OBJ file")
        for vertex in vertices:
            obj_content.append("v {} {} {}".format(
                vertex[0], vertex[1], vertex[2]))
        for face in faces:
            obj_content.append("f {} {} {}".format(
                face[0] + 1, face[1] + 1, face[2] + 1))

        # Write the OBJ file
        with open(filename, 'w') as obj_file:
            obj_file.write("\n".join(obj_content))
        
        logger.info(f"OBJ file saved successfully: {filename}")

    def get_verticesGT(self):
        return self.verticesGT

    def get_facesGT(self):
        return self.facesGT

    def get_classesGT(self):
        return self.classesGT

    def get_verticesPre(self):
        return self.verticesPre

    def get_facesPre(self):
        return self.facesPre

    def get_classesPre(self):
        return self.classesPre
