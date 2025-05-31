import SimpleITK as sitk
import numpy as np
import os # For file existence check, though SimpleITK also handles it

def read_mha_image(file_path: str) -> dict:
    """
    Reads an MHA (MetaImage) file and extracts image data and metadata.

    Args:
        file_path (str): The path to the MHA file.

    Returns:
        dict: A dictionary containing the image data as a NumPy array
              and essential metadata (dimensions, spacing, origin, direction).
              Returns None if an error occurs.
    """
    try:
        # Check if file exists (optional, SimpleITK ReadImage throws an error too)
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'")
            return None

        # 1. Read the MHA file
        image = sitk.ReadImage(file_path)

        # 2. Extract image data as a NumPy array
        # Note: SimpleITK's GetArrayFromImage returns array with order [z,y,x] for 3D
        image_data = sitk.GetArrayFromImage(image)

        # 3. Extract essential metadata
        dimensions = list(image.GetSize())         # [nx, ny, nz]
        spacing = list(image.GetSpacing())         # [sx, sy, sz]
        origin = list(image.GetOrigin())           # [ox, oy, oz]

        # Direction Cosine Matrix (DCM)
        # GetDirection returns a flattened tuple (e.g., 9 elements for 3D)
        direction_flat = image.GetDirection()
        num_dims = image.GetDimension()
        if num_dims > 0 and len(direction_flat) == num_dims * num_dims:
            direction_matrix = np.array(direction_flat).reshape(num_dims, num_dims).tolist()
        else:
            # Handle cases where direction might not be set or is unexpected
            direction_matrix = None

        return {
            'data': image_data,
            'dimensions': dimensions,
            'spacing': spacing,
            'origin': origin,
            'direction': direction_matrix
        }

    except RuntimeError as e:
        print(f"Error reading or processing MHA file '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with file '{file_path}': {e}")
        return None

if __name__ == '__main__':
    print("Testing MHA image reader function...\n")

    # Create a dummy MHA file for testing purposes
    # This is a very basic MHA header, actual MHA files have more metadata.
    dummy_mha_file_path = "dummy_test.mha"
    dummy_raw_file_path = "dummy_test.raw"

    # Create a small dummy raw file (e.g., 2x2x2 unsigned char)
    try:
        # Create a small 3D numpy array
        dummy_data_shape = (2, 3, 4) # z, y, x
        dummy_np_array = np.arange(np.prod(dummy_data_shape), dtype=np.uint8).reshape(dummy_data_shape)

        # Write to a raw file (binary)
        with open(dummy_raw_file_path, 'wb') as f_raw:
            f_raw.write(dummy_np_array.tobytes())

        mha_header_content = f"""ObjectType = Image
NDims = 3
DimSize = {' '.join(map(str, reversed(dummy_data_shape)))}
ElementType = MET_UCHAR
ElementSpacing = 1.0 1.0 1.0
Offset = 0.0 0.0 0.0
BinaryData = True
BinaryDataByteOrderMSB = False
ElementDataFile = {os.path.basename(dummy_raw_file_path)}
"""
        with open(dummy_mha_file_path, 'w') as f_mha:
            f_mha.write(mha_header_content)

        print(f"Created dummy MHA file: '{dummy_mha_file_path}' for testing.")
        print(f"Created dummy RAW file: '{dummy_raw_file_path}' with shape {dummy_data_shape} (z,y,x for numpy).\n")

        image_info = read_mha_image(dummy_mha_file_path)

        if image_info:
            print(f"Successfully read MHA file: '{dummy_mha_file_path}'")
            print("Keys in returned dictionary:", list(image_info.keys()))

            print("\n--- Metadata ---")
            print(f"Dimensions (Size): {image_info['dimensions']}") # Should be [4, 3, 2] (x,y,z)
            print(f"Spacing: {image_info['spacing']}")
            print(f"Origin (Offset): {image_info['origin']}")
            print(f"Direction Matrix: {image_info['direction']}") # Default is identity

            print("\n--- Data ---")
            if isinstance(image_info['data'], np.ndarray):
                print(f"Data array shape: {image_info['data'].shape}") # Should be [2, 3, 4] (z,y,x)
                print(f"Data array dtype: {image_info['data'].dtype}")
                # Verify some data if possible
                if image_info['data'].size > 0:
                    print(f"First element of data array: {image_info['data'].ravel()[0]}")
                    print(f"Last element of data array: {image_info['data'].ravel()[-1]}")

                # Compare with original dummy data
                # Note: SimpleITK GetArrayFromImage gives [z,y,x], DimSize in MHA is [x,y,z]
                # Our dummy_np_array is already [z,y,x]
                if np.array_equal(image_info['data'], dummy_np_array):
                    print("Read data matches original dummy NumPy array.")
                else:
                    print("Mismatch between read data and original dummy NumPy array!") # pragma: no cover
            else:
                print("Data is not a NumPy array.") # pragma: no cover
        else:
            print(f"Failed to read MHA file: '{dummy_mha_file_path}'") # pragma: no cover

    except Exception as e:
        print(f"Error in example usage: {e}") # pragma: no cover
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_mha_file_path):
            os.remove(dummy_mha_file_path)
        if os.path.exists(dummy_raw_file_path):
            os.remove(dummy_raw_file_path)
        print("\nCleaned up dummy files.")

    print("\n--- Test with a non-existent file ---")
    non_existent_file = "non_existent_file.mha"
    image_info_non_existent = read_mha_image(non_existent_file)
    if image_info_non_existent is None:
        print(f"Correctly handled non-existent file: '{non_existent_file}' (returned None).")
    else:
        print(f"Incorrectly handled non-existent file: '{non_existent_file}'.") # pragma: no cover

    print("\n--- Test with an invalid MHA (e.g., an empty file) ---")
    invalid_mha_file_path = "invalid.mha"
    with open(invalid_mha_file_path, 'w') as f:
        f.write("This is not a valid MHA file.")

    image_info_invalid = read_mha_image(invalid_mha_file_path)
    if image_info_invalid is None:
        print(f"Correctly handled invalid MHA file: '{invalid_mha_file_path}' (returned None).")
    else:
        print(f"Incorrectly handled invalid MHA file: '{invalid_mha_file_path}'.") # pragma: no cover

    if os.path.exists(invalid_mha_file_path):
        os.remove(invalid_mha_file_path)
    print("Cleaned up invalid MHA file.")

    print("\nEnd of MHA reader test.")
