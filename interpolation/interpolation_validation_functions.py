import numpy as np
import os
import tensorflow as tf
import imageio
#import imageio.v2 as imageio
from PIL import Image
from natsort import natsorted
#import cv2
# import mahotas
#import pandas as pd
#import csv
# from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None

def load_image(img_path: str):
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    image = imageio.imread(img_path)
    image = image.astype(np.float32) / _UINT8_MAX_F
    return image

def get_file_extension(filename):
    return os.path.splitext(filename)[1]

def pad_and_tile_image(image: np.ndarray, tile_size: tuple):
    h, w, _ = image.shape
    tile_height, tile_width = tile_size
    pad_h = (tile_height - h % tile_height) % tile_height
    pad_w = (tile_width - w % tile_width) % tile_width
    padded_image = np.pad(image, pad_width=[(0, pad_h), (0, pad_w), (0, 0)], mode='constant', constant_values=0)
    padded_height, padded_width, _ = padded_image.shape
    tiled_array = padded_image.reshape(padded_height // tile_height,
                                       tile_height,
                                       padded_width // tile_width,
                                       tile_width,
                                       -1)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array, (pad_h, pad_w)

def stitch_tiles(tiles: np.ndarray, pad_h: int, pad_w: int, tile_size: tuple):
    tile_height, tile_width = tile_size
    tile_rows, tile_cols, _, _, _ = tiles.shape
    stitched_image = tiles.swapaxes(1, 2).reshape(tile_rows * tile_height, tile_cols * tile_width, -1)
    if pad_h > 0 or pad_w > 0:
        stitched_image = stitched_image[:-pad_h or None, :-pad_w or None]

    return stitched_image


# def interpolate_images_validation(pth, skips, TILE_SIZE, model):
#     """
#     Function to interpolate images from a directory with skip intervals.
#
#     Args:
#         pth (str): Path to the directory containing images.
#         skips (list): List of skip numbers for generating interpolation times.
#         TILE_SIZE (tuple): Size of the tiles to split the images into.
#         model: The trained model used for interpolation.
#     """
#     _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
#     image_files = [_ for _ in os.listdir(pth) if _.endswith(('tif', 'png', 'jpg'))]
#     image_files = natsorted(image_files)
#     file_extension = get_file_extension(image_files[0])
#
#     for skip_num in skips:
#         print(f'Interpolating int {skip_num}:')
#         output_folder = os.path.join(pth, f'skip_{skip_num}')
#         os.makedirs(output_folder, exist_ok=True)
#         times = np.linspace(0, 1, skip_num + 2)
#         times = times[1:-1]
#
#         loaded_images = image_files[::skip_num + 1]
#         impairs = zip(loaded_images[:-1], loaded_images[1:])
#         imtargets = [image_files[i + 1:i + skip_num + 1] for i in range(0, len(image_files), skip_num + 1)]
#
#         for impair, imtarget in zip(impairs, imtargets):
#
#             pthim1 = os.path.join(pth, impair[0])
#             pthim2 = os.path.join(pth, impair[1])
#             output_checkpath = os.path.join(output_folder, f"{os.path.splitext(imtarget[0])[0]}{file_extension}")
#             if os.path.exists(output_checkpath):
#                 print(f'already interpolated from input images {impair[0]} and {impair[1]}')
#                 continue
#
#             image1 = load_image(pthim1)
#             image2 = load_image(pthim2)
#
#             if max(image1.shape[:2]) > 1024 or max(image2.shape[:2]) > 1024:
#                 tiles1, (pad_h, pad_w) = pad_and_tile_image(image1, TILE_SIZE)
#                 tiles2, _ = pad_and_tile_image(image2, TILE_SIZE)
#                 tile_height, tile_width = TILE_SIZE
#                 num_channels = image1.shape[-1]
#                 tile_rows, tile_cols, _, _, _ = tiles1.shape
#             else:
#                 for idx, time in enumerate(times):
#                     filename = f"{os.path.splitext(imtarget[idx])[0]}{file_extension}"
#                     output_path = os.path.join(output_folder, filename)
#                     position = next((i for i, targets in enumerate(imtargets) if imtarget[idx] in targets), None)
#                     print(f'Interpolating {filename}  ({position + 1} /{len(imtargets) - 1})')
#
#                     if os.path.exists(output_path):
#                         print(f'already interpolated from input images {impair[0]} and {impair[1]}')
#                         continue
#                     time1 = np.array([time], dtype=np.float32)
#                     input_data = {
#                         'time': np.array([time1], dtype=np.float32),
#                         'x0': np.expand_dims(image1, axis=0),
#                         'x1': np.expand_dims(image2, axis=0)
#                     }
#                     mid_frame = model(input_data)
#                     generated_image = mid_frame['image'][0].numpy()
#                     image_in_uint8_range = np.clip(generated_image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
#                     image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
#
#                     imageio.imwrite(output_path, image_in_uint8, format=file_extension.lstrip('.'))
#                 continue
#             tiles_dict = {i: [] for i in range(len(imtarget))}
#
#             for idx, time in enumerate(times):
#                 for tile_row in range(tile_rows):
#                     for tile_col in range(tile_cols):
#                         tile1 = tiles1[tile_row, tile_col]
#                         tile2 = tiles2[tile_row, tile_col]
#
#                         time1 = np.array([time], dtype=np.float32)
#                         input_data = {
#                             'time': np.array([time1], dtype=np.float32),
#                             'x0': np.expand_dims(tile1, axis=0),
#                             'x1': np.expand_dims(tile2, axis=0)
#                         }
#                         mid_frame = model(input_data)
#                         generated_tile = mid_frame['image'][0].numpy()
#                         image_in_uint8_range = np.clip(generated_tile * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
#                         image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
#
#                         tiles_dict[idx].append(image_in_uint8)
#
#             for idx, tiles in tiles_dict.items():
#                 stitched_filename = f"{os.path.splitext(imtarget[idx])[0]}{file_extension}"
#                 stitched_output_path = os.path.join(output_folder, stitched_filename)
#                 position = next((i for i, targets in enumerate(imtargets) if imtarget[idx] in targets), None)
#                 print(f'Interpolating {stitched_filename}  ({position + 1} /{len(imtargets) - 1})')
#
#                 #num_tiles = len(tiles)
#
#                 interpolated_tiles_np = np.array(tiles).reshape(tile_rows, tile_cols, tile_height, tile_width,
#                                                                 num_channels)
#                 stitched_image = stitch_tiles(interpolated_tiles_np, pad_h, pad_w, TILE_SIZE)
#
#                 imageio.imwrite(stitched_output_path, stitched_image, format=file_extension.lstrip('.'))

# def interpolate_images_validation(pth, skips, TILE_SIZE, model):
#     _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
#     image_files = [_ for _ in os.listdir(pth) if _.endswith(('tif', 'png', 'jpg'))]
#     image_files = natsorted(image_files)
#     file_extension = get_file_extension(image_files[0])
#
#     image2 = None  # Avoid reloading the same image
#
#     for skip_num in skips:
#         print(f'Interpolating int {skip_num}:')
#         output_folder = os.path.join(pth, f'skip_{skip_num}')
#         os.makedirs(output_folder, exist_ok=True)
#         times = np.linspace(0, 1, skip_num + 2)[1:-1]
#
#         loaded_images = image_files[::skip_num + 1]
#         impairs = zip(loaded_images[:-1], loaded_images[1:])
#         imtargets = [image_files[i + 1:i + skip_num + 1] for i in range(0, len(image_files), skip_num + 1)]
#
#         for impair, imtarget in zip(impairs, imtargets):
#
#             pthim1 = os.path.join(pth, impair[0])
#             pthim2 = os.path.join(pth, impair[1])
#             output_checkpath = os.path.join(output_folder, f"{os.path.splitext(imtarget[0])[0]}{file_extension}")
#
#             if os.path.exists(output_checkpath):
#                 print(f'already interpolated from input images {impair[0]} and {impair[1]}')
#                 continue
#
#             if image2 is not None:
#                 print('loading image 1 from image 2')
#                 image1 = image2
#             else:
#                 image1 = load_image(pthim1)
#
#             image2 = load_image(pthim2)
#
#             if max(image1.shape[:2]) > TILE_SIZE[0] or max(image2.shape[:2]) > TILE_SIZE[1]:
#                 tiles1, (pad_h, pad_w) = pad_and_tile_image(image1, TILE_SIZE)
#                 tiles2, _ = pad_and_tile_image(image2, TILE_SIZE)
#                 tile_height, tile_width = TILE_SIZE
#                 num_channels = image1.shape[-1]
#                 tile_rows, tile_cols, _, _, _ = tiles1.shape
#             else:
#                 with tqdm(total=len(times), desc=f"Interpolating {impair[0]}", unit="frame") as pbar:
#                     for idx, time in enumerate(times):
#                         start_time = time.time()
#                         filename = f"{os.path.splitext(imtarget[idx])[0]}{file_extension}"
#                         output_path = os.path.join(output_folder, filename)
#
#                         if os.path.exists(output_path):
#                             print(f'already interpolated from input images {impair[0]} and {impair[1]}')
#                             pbar.update(1)
#                             continue
#
#                         time1 = np.array([time], dtype=np.float32)
#                         input_data = {
#                             'time': np.array([time1], dtype=np.float32),
#                             'x0': np.expand_dims(image1, axis=0),
#                             'x1': np.expand_dims(image2, axis=0)
#                         }
#                         mid_frame = model(input_data)
#                         generated_image = mid_frame['image'][0].numpy()
#                         image_in_uint8_range = np.clip(generated_image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
#                         image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
#
#                         imageio.imwrite(output_path, image_in_uint8, format=file_extension.lstrip('.'))
#                         elapsed_time = time.time() - start_time
#                         print(f"Time to generate {filename}: {elapsed_time:.2f} seconds")
#                         pbar.update(1)
#                 continue
#
#             with tqdm(total=len(times), desc=f"Interpolating {impair[0]}", unit="frame") as pbar:
#                 for idx, time in enumerate(times):
#                     start_time = time.time()
#                     tiles_dict = []  # Changed from dict to list to avoid memory growth
#
#                     for tile_row in range(tile_rows):
#                         for tile_col in range(tile_cols):
#                             tile1 = tiles1[tile_row, tile_col]
#                             tile2 = tiles2[tile_row, tile_col]
#
#                             time1 = np.array([time], dtype=np.float32)
#                             input_data = {
#                                 'time': np.array([time1], dtype=np.float32),
#                                 'x0': np.expand_dims(tile1, axis=0),
#                                 'x1': np.expand_dims(tile2, axis=0)
#                             }
#                             mid_frame = model(input_data)
#                             generated_tile = mid_frame['image'][0].numpy()
#                             image_in_uint8_range = np.clip(generated_tile * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
#                             image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
#
#                             tiles_dict.append(image_in_uint8)
#
#                     stitched_filename = f"{os.path.splitext(imtarget[idx])[0]}{file_extension}"
#                     stitched_output_path = os.path.join(output_folder, stitched_filename)
#
#                     interpolated_tiles_np = np.array(tiles_dict).reshape(tile_rows, tile_cols, tile_height, tile_width, num_channels)
#                     stitched_image = stitch_tiles(interpolated_tiles_np, pad_h, pad_w, TILE_SIZE)
#
#                     imageio.imwrite(stitched_output_path, stitched_image, format=file_extension.lstrip('.'))
#
#                     elapsed_time = time.time() - start_time
#                     print(f"Time to generate {stitched_filename}: {elapsed_time:.2f} seconds")
#                     pbar.update(1)
def interpolate_skipped_images(pthims, TILE_SIZE, model):
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    image_files = [f for f in os.listdir(pthims) if f.endswith(('tif', 'png', 'jpg'))]
    image_files = natsorted(image_files)
    file_extension = get_file_extension(image_files[0])
    image2 = None

    output_folder = os.path.join(pthims, "skipped_interpolations")
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(image_files) - 2):  # Skip one image
        image1_path = os.path.join(pthims, image_files[i])
        skipped_image_path = os.path.join(pthims, image_files[i + 1])
        image2_path = os.path.join(pthims, image_files[i + 2])

        output_checkpath = os.path.join(output_folder, image_files[i + 1])
        if os.path.exists(output_checkpath):
            print(f'Already interpolated: {image_files[i + 1]}')
            continue

        if image2 is not None:
            print('Loading image 1 from previous image 2')
            image1 = image2
        else:
            image1 = load_image(image1_path)

        image2 = load_image(image2_path)

        print(f'Interpolating {image_files[i + 1]} from {image_files[i]} and {image_files[i + 2]}')

        time_value = np.array([0.5], dtype=np.float32)  # Middle point for interpolation

        if max(image1.shape[:2]) > TILE_SIZE[0] or max(image2.shape[:2]) > TILE_SIZE[1]:
            print('Using tiling for interpolation')
            tiles1, (pad_h, pad_w) = pad_and_tile_image(image1, TILE_SIZE)
            tiles2, _ = pad_and_tile_image(image2, TILE_SIZE)
            tile_height, tile_width = TILE_SIZE
            tile_rows, tile_cols, _, _, _ = tiles1.shape

            tiles_dict = []
            for tile_row in range(tile_rows):
                for tile_col in range(tile_cols):
                    tile1 = tiles1[tile_row, tile_col]
                    tile2 = tiles2[tile_row, tile_col]

                    input_data = {
                        'time': np.array([time_value], dtype=np.float32),
                        'x0': np.expand_dims(tile1, axis=0),
                        'x1': np.expand_dims(tile2, axis=0)
                    }
                    mid_frame = model(input_data)
                    generated_tile = mid_frame['image'][0].numpy()
                    image_in_uint8_range = np.clip(generated_tile * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
                    tiles_dict.append((image_in_uint8_range + 0.5).astype(np.uint8))

            interpolated_tiles_np = np.array(tiles_dict).reshape(tile_rows, tile_cols, tile_height, tile_width, 3)
            stitched_image = stitch_tiles(interpolated_tiles_np, pad_h, pad_w, TILE_SIZE)
            imageio.imwrite(output_checkpath, stitched_image, format=file_extension.lstrip('.'))
        else:
            input_data = {
                'time': np.array([time_value], dtype=np.float32),
                'x0': np.expand_dims(image1, axis=0),
                'x1': np.expand_dims(image2, axis=0)
            }
            mid_frame = model(input_data)
            generated_image = mid_frame['image'][0].numpy()
            image_in_uint8_range = np.clip(generated_image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
            image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
            imageio.imwrite(output_checkpath, image_in_uint8, format=file_extension.lstrip('.'))

        print(f'Generated {image_files[i + 1]}')

#
# def interpolate_images_validation(pth, skips, TILE_SIZE, model):
#     _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
#     image_files = [_ for _ in os.listdir(pth) if _.endswith(('tif', 'png', 'jpg'))]
#     image_files = natsorted(image_files)
#     file_extension = get_file_extension(image_files[0])
#
#     for skip_num in skips:
#         # Create output folder
#         output_folder = os.path.join(pth, f'skip_{skip_num}')
#         os.makedirs(output_folder, exist_ok=True)
#
#         # Generate interpolation times
#         times = np.linspace(0, 1, skip_num + 2)[1:-1]
#
#         # Process images in pairs
#         for i in range(len(image_files) - 1):
#             image1_path = os.path.join(pth, image_files[i])
#             image2_path = os.path.join(pth, image_files[i + 1])
#
#             image1 = load_image(image1_path)
#             image2 = load_image(image2_path)
#
#             # TILE
#             if max(image1.shape[:2]) > TILE_SIZE[0] or max(image2.shape[:2]) > TILE_SIZE[1]:
#                 # Pad and split images into tiles if they are larger than TILE_SIZE
#                 tiles1, (pad_h, pad_w) = pad_and_tile_image(image1, TILE_SIZE)
#                 tiles2, _ = pad_and_tile_image(image2, TILE_SIZE)
#             else:
#                 # If images are smaller or equal to TILE_SIZE, treat the whole image as a single tile
#                 tiles1 = np.expand_dims(image1, axis=0)
#                 tiles2 = np.expand_dims(image2, axis=0)
#                 pad_h = pad_w = 0
#
#             tile_height, tile_width = TILE_SIZE
#             num_channels = image1.shape[-1]
#             tile_rows, tile_cols, _, _, _ = tiles1.shape
#
#             for idx, time in enumerate(times):
#                 tiles_dict = []
#                 for tile_row in range(tile_rows):
#                     for tile_col in range(tile_cols):
#                         tile1 = tiles1[tile_row, tile_col]
#                         tile2 = tiles2[tile_row, tile_col]
#
#                         time1 = np.array([time], dtype=np.float32)
#                         input_data = {
#                             'time': np.array([time1], dtype=np.float32),
#                             'x0': np.expand_dims(tile1, axis=0),
#                             'x1': np.expand_dims(tile2, axis=0)
#                         }
#                         mid_frame = model(input_data)
#                         generated_tile = mid_frame['image'][0].numpy()
#                         image_in_uint8_range = np.clip(generated_tile * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
#                         image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)
#
#                         # Append tile to list
#                         tiles_dict.append(image_in_uint8)
#
#                 # Convert the list of tiles back into a numpy array with the original tile shape
#                 interpolated_tiles_np = np.array(tiles_dict).reshape(tile_rows, tile_cols, tile_height, tile_width,
#                                                                      num_channels)
#
#                 # Stitch the tiles back together
#                 stitched_image = stitch_tiles(interpolated_tiles_np, pad_h, pad_w, TILE_SIZE)
#
#                 # Generate filename for the stitched image
#                 stitched_filename = f"{os.path.splitext(image_files[i])[0]}_int{idx + 1}{file_extension}"
#                 stitched_output_path = os.path.join(output_folder, stitched_filename)
#
#                 # Save the stitched image to the output folder
#                 imageio.imwrite(stitched_output_path, stitched_image, format=file_extension.lstrip('.'))
#
# def calculate_haralick(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     haralick_features = mahotas.features.haralick(image).mean(axis=0)
#     return haralick_features
#
# # def process_images_in_folder(folder_path, output_csv):
# #     image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.jpg', '.png'))]
# #     with open(output_csv, 'w', newline='') as csv_file:
# #         writer = csv.writer(csv_file)
# #         header = ['Image'] + [f'Haralick_{i}' for i in range(13)]  # 13 Haralick features
# #         writer.writerow(header)
# #         for image_file in image_files:
# #             image_path = os.path.join(folder_path, image_file)
# #             haralick_features = calculate_haralick(image_path)
# #             row = [image_file] + list(haralick_features)
# #             writer.writerow(row)
# def process_images_in_folder(folder_path, output_csv):
#     image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.jpg', '.png'))]
#     csv_folder= os.path.join(folder_path,'haralick_features_quantifications')
#     os.makedirs(csv_folder, exist_ok=True)
#     csv_path = os.path.join(csv_folder, output_csv)  # Save CSV in the same directory as images
#     with open(csv_path, 'w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         header = ['Image'] + [f'Haralick_{i}' for i in range(13)]  # 13 Haralick features
#         writer.writerow(header)
#         for image_file in image_files:
#             image_path = os.path.join(folder_path, image_file)
#             haralick_features = calculate_haralick(image_path)
#             row = [image_file] + list(haralick_features)
#             writer.writerow(row)
#
# def process_interpolated_images_for_haralick_features(pth, skips):
#     """
#     Function to process images in folders generated by interpolation for Haralick features.
#
#     Args:
#         pth (str): Path to the directory containing images.
#         skips (list): List of skip numbers corresponding to the generated middle image folders.
#     """
#
#     for skip_num in skips:
#         # Specify the output folder for the generated middle images
#         # output_folder = os.path.join(pth, f'skip_{skip_num}')
#         outnm = f'skip_{skip_num}_haralick_features.csv'
#
#         # Process the images in the output folder and save Haralick features to a CSV file
#         process_images_in_folder(pth, outnm)