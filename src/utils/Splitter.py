# This module provides a Splitter class for processing large images by splitting them into smaller chunks and reassembling results.

import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from concurrent.futures import ThreadPoolExecutor
from .viz import display_grayscale, display_edges

class Splitter:
    """Process large images by splitting into smaller chunks, applying a mask function, and reassembling results."""
    
    # The splitter is used to split and image, process it by chunks, and return a mask or image of the same shape as output
    # It assumes the mask_function must take an input of fixed size sub_image_size * sub_image_size
    
    # join function takes two arrays of the same shape and outputs on array
    # mask_function takes a numpy array and returns a numpy array of the same shape
    # if batch size is not 1, batch_mask_function is used in get_full_masks to accelerate with input being a batch
    # batch size only for acceleration, used in the get_full_masks only
    
    def __init__(self, sub_image_size, mask_function, join_function = (lambda x,y : np.where(x == 0, y, x)), 
                 mask_dtype=np.float32, batch_size=1, batch_mask_function=None, invalid_fill_value=np.nan,
                 max_workers=None):
        """Initialize splitter with processing parameters and functions.
        
        Args:
            max_workers: Maximum number of threads for parallel processing in get_masks().
                        None (default) = sequential processing. Set to enable threading (max 10 threads).
        """
        self.sub_image_size = sub_image_size
        self.mask_function = mask_function
        self.join_function = join_function
        self.mask_dtype=mask_dtype
        
        self.invalid_fill_value = invalid_fill_value
        
        
        self.batch_size = batch_size
        self.batch_mask_function = batch_mask_function
        self.max_workers = max_workers
    
    def get_mask(self, sub_image):
        """Apply mask function to a single sub-image."""
        a = self.mask_function(sub_image)
        return a
    
    def get_masks(self, sub_images):
        """Apply mask function to multiple sub-images sequentially or in parallel."""
        if len(sub_images) == 0:
            return np.array([], dtype=self.mask_dtype)
        
        # Pre-allocate output array for efficiency (Issue 1)
        first_mask = self.get_mask(sub_images[0])
        output_shape = (len(sub_images),) + first_mask.shape
        masks = np.empty(output_shape, dtype=self.mask_dtype)
        masks[0] = first_mask
        
        # Use threading if enabled (max 10 threads)
        if len(sub_images) > 1 and self.max_workers is not None:
            max_workers = min(self.max_workers, 10, len(sub_images) - 1)  # Cap at 10 threads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self.get_mask, sub_images[1:]))
            masks[1:] = results
        else:
            # Sequential processing
            for i in range(1, len(sub_images)):
                masks[i] = self.get_mask(sub_images[i])
        
        return masks
    
    def get_mask_batch(self, batch):
        """Apply batch mask function to a batch of sub-images."""
        a = self.batch_mask_function(batch)
        return a
    
    def get_masks_in_batch(self, sub_images, batch_size):
        """Process sub-images in batches for acceleration."""
        n_images = len(sub_images)
        if n_images == 0:
            return np.array([], dtype=self.mask_dtype)
        
        # Process first batch to determine output shape (Issue 2)
        first_batch_size = min(batch_size, n_images)
        first_batch = sub_images[0:first_batch_size]
        first_masks = self.get_mask_batch(first_batch).astype(self.mask_dtype, copy=False)
        
        if n_images <= batch_size:
            return first_masks
        
        # Pre-allocate output array to avoid intermediate lists and concatenations
        output_shape = (n_images,) + first_masks.shape[1:]
        all_masks = np.empty(output_shape, dtype=self.mask_dtype)
        all_masks[0:first_batch_size] = first_masks
        
        # Process remaining full batches
        idx = first_batch_size
        # Calculate the last index that starts a full batch
        last_full_batch_start = (n_images // batch_size) * batch_size
        for i in range(batch_size, last_full_batch_start, batch_size):
            batch = sub_images[i:i+batch_size]
            masks = self.get_mask_batch(batch).astype(self.mask_dtype, copy=False)
            batch_len = len(masks)
            all_masks[idx:idx+batch_len] = masks
            idx += batch_len
        
        # Process remaining items individually
        for i in range(idx, n_images):
            mask = self.get_mask(sub_images[i]).astype(self.mask_dtype, copy=False)
            all_masks[i] = mask
        
        return all_masks
        
    def pad_right_down(self, arr, target_shape):
        """Pad array to target shape by adding zeros to the right and bottom."""
        pad_width = []
        for current, target in zip(arr.shape, target_shape):
            pad_before = 0
            pad_after = max(target - current, 0)
            pad_width.append((pad_before, pad_after))
        
        return np.pad(arr, pad_width, mode='constant', constant_values=self.invalid_fill_value)
    
    def split_1D(self, column):
        """Split a 1D column into chunks of sub_image_size."""
        num_full_chunks = column.shape[0] // self.sub_image_size
        trimmed = column[:num_full_chunks * self.sub_image_size, :]
        return np.split(trimmed, num_full_chunks, axis=0)

    def get_full_masks(self, image):
        """Process image by splitting into windows and applying mask function."""
        window_shape = list(image.shape)
        window_shape[0] = self.sub_image_size
        window_shape[1] = self.sub_image_size
        window_shape_tup = tuple(window_shape)
        
        step = [1 for _ in range(len(image.shape))]
        step[0] = self.sub_image_size
        step[1] = self.sub_image_size
        step_tup = tuple(step)
                
        images = view_as_windows(image, window_shape=window_shape_tup, step=step_tup)
        axes_to_squeeze = [i for i in range(images.ndim - 1) if images.shape[i] == 1]
        images = images.squeeze(axis=tuple(axes_to_squeeze))
        rows, cols = images.shape[:2]
        normal_shape = [rows, cols] + [self.sub_image_size, self.sub_image_size]
        
        split_images = images.reshape(-1, self.sub_image_size, self.sub_image_size, *image.shape[2:])
    
        
        # split_images is already a numpy array after reshape, no need to convert (Issue 3)
        images = split_images
        if self.batch_size == 1:
            split_masks = self.get_masks(images)
        else: 
            split_masks = self.get_masks_in_batch(images, batch_size=self.batch_size)
        # expect output masks to have shape 
        output_dims = list(split_masks.shape)
        output_dims.remove(rows * cols)
        output_dims.remove(self.sub_image_size)
        output_dims.remove(self.sub_image_size)
        
        normal_shape = tuple(normal_shape + output_dims)
        new_split_masks = split_masks.reshape(normal_shape)
           
        axes = [0, 2, 1] + list(range(3, len(new_split_masks.shape)))
        new_split_masks = new_split_masks.transpose(axes)
        combined_shape = [rows * self.sub_image_size, cols * self.sub_image_size] + output_dims
        large_mask = new_split_masks.reshape(combined_shape)   
        
        large_mask = self.pad_right_down(large_mask, [image.shape[0], image.shape[1]] + output_dims)  
        return large_mask.astype(self.mask_dtype, copy=False)

    def process_1D(self, col):
        """Process a 1D column by splitting and applying mask function."""
        other_dims = list(col.shape)[2:]
        split = np.array(self.split_1D(col)).reshape([-1,self.sub_image_size,self.sub_image_size] + other_dims)
        a = self.get_masks(split)
        if a.shape[1] == 1: a = a.squeeze(axis=1)
        output_other_dims = list(a.shape)[3:]
        split_masks = a.reshape([-1,self.sub_image_size,self.sub_image_size] + output_other_dims)
        combined_mask = a.reshape([-1,self.sub_image_size] + output_other_dims)
        a = self.pad_right_down(combined_mask, tuple([col.shape[0], self.sub_image_size] + output_other_dims))
        return a
        
    def get_rightmost_trace(self, img, result_shape):
        """Process the rightmost edge of the image."""
        total_trace = np.full(result_shape, self.invalid_fill_value ,dtype=self.mask_dtype)
        total_trace[:, -self.sub_image_size:] = self.process_1D(img[:, -self.sub_image_size:])
        # display_grayscale(total_trace, title="get_rightmost_trace")
        return total_trace
    
    def get_downmost_trace(self, img, result_shape):
        """Process the bottom edge of the image."""
        total_trace = np.full(result_shape, self.invalid_fill_value, dtype=self.mask_dtype)
        other_dims = list(range(2, len(img.shape)))
        a = self.process_1D(img[-self.sub_image_size:, :].transpose([1, 0] + other_dims))
        total_trace[-self.sub_image_size:, :] = a.transpose([1, 0] + other_dims)
        # display_grayscale(total_trace, title="get_downmost_trace")
        return total_trace

    
    def get_bottom_right_corner_trace(self, img, result_shape):
        """Process the bottom-right corner of the image."""
        other_dims = list(img.shape)[2:]
        total_trace = np.full(result_shape, self.invalid_fill_value ,dtype=self.mask_dtype)
        cell = img[-self.sub_image_size:, -self.sub_image_size:].reshape([self.sub_image_size,self.sub_image_size] + other_dims)
        total_trace[-self.sub_image_size:, -self.sub_image_size:] = self.get_mask(cell)
        return total_trace
    

    def process(self, image):
        """Process the entire image by splitting, applying mask function, and reassembling."""
        if len(image.shape) == (self.sub_image_size,self.sub_image_size):
            return self.get_mask(image)
    
        full_mask = self.get_full_masks(image)
        full_mask = self.join_function(full_mask, self.get_rightmost_trace(image, result_shape=full_mask.shape))
        full_mask = self.join_function(full_mask, self.get_downmost_trace(image, result_shape=full_mask.shape))
        return self.join_function(full_mask, self.get_bottom_right_corner_trace(image, result_shape=full_mask.shape))
           
    
    

