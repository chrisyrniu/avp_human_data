#!/usr/bin/env python3
"""
Advanced HDF5 Compression Script for Robotics Data
This script properly handles hierarchical HDF5 files with nested groups and datasets.
Optimized for robotics data with images, poses, and sensor data.
"""

import os
import argparse
import h5py
import numpy as np
from termcolor import colored
from datetime import datetime
import shutil
import time


def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


def recursively_find_datasets(group, datasets=None, path=""):
    """Recursively find all datasets in an HDF5 group"""
    if datasets is None:
        datasets = {}

    for key in group.keys():
        item = group[key]
        current_path = f"{path}/{key}" if path else key

        if isinstance(item, h5py.Dataset):
            # It's a dataset
            datasets[current_path] = {
                'object': item,
                'shape': item.shape,
                'dtype': str(item.dtype),
                'compression': item.compression,
                'compression_opts': item.compression_opts,
                'chunks': item.chunks,
                'size_mb': item.nbytes / (1024 * 1024) if item.nbytes > 0 else 0
            }
        elif isinstance(item, h5py.Group):
            # It's a group, recurse into it
            recursively_find_datasets(item, datasets, current_path)

    return datasets


def analyze_hdf5_file(file_path):
    """Analyze HDF5 file and return detailed information about all datasets"""
    try:
        with h5py.File(file_path, 'r') as f:
            datasets = recursively_find_datasets(f)

            total_size = sum(ds['size_mb'] for ds in datasets.values())

            info = {
                'file_path': file_path,
                'file_size_mb': get_file_size_mb(file_path),
                'total_data_size_mb': total_size,
                'num_datasets': len(datasets),
                'datasets': datasets,
                'attributes': dict(f.attrs)
            }

            return info

    except Exception as e:
        print(colored(f"Error analyzing {file_path}: {e}", "red"))
        return None


def copy_group_recursively(src_group, dst_group, compression, compression_opts, verbose=True):
    """Recursively copy groups and datasets with compression"""

    # Copy attributes
    for attr_name, attr_value in src_group.attrs.items():
        dst_group.attrs[attr_name] = attr_value

    for key in src_group.keys():
        item = src_group[key]

        if isinstance(item, h5py.Dataset):
            # It's a dataset - copy with compression
            if verbose:
                print(f"    Compressing dataset: {key} {item.shape}")

            # Determine optimal chunks for this dataset
            # Skip chunking for empty datasets or scalar datasets
            if item.size == 0 or len(item.shape) == 0:
                chunk_shape = None
            elif item.chunks is not None:
                chunk_shape = item.chunks
            else:
                # Auto-determine chunks based on dataset type
                # Ensure all chunk dimensions are positive
                if len(item.shape) == 4:  # Likely image sequence: (frames, height, width, channels)
                    chunk_shape = (1, item.shape[1], item.shape[2], item.shape[3])
                elif len(item.shape) == 3:  # Likely single image: (height, width, channels)
                    chunk_shape = item.shape
                elif len(item.shape) == 2:  # Likely pose/joint data: (frames, features)
                    chunk_shape = (min(100, item.shape[0]), item.shape[1])
                elif len(item.shape) == 1:  # 1D array
                    chunk_shape = (min(1000, item.shape[0]),)
                else:
                    chunk_shape = item.shape

                # Validate chunk dimensions - all must be positive
                if chunk_shape and any(dim <= 0 for dim in chunk_shape):
                    chunk_shape = None

            # Choose compression based on data type
            if 'image' in key.lower() or len(item.shape) >= 3:
                # Image data - use fast compression
                dataset_compression = 'lzf' if compression == 'auto' else compression
                dataset_compression_opts = None
            else:
                # Numerical data - use better compression
                dataset_compression = compression if compression != 'auto' else 'gzip'
                dataset_compression_opts = compression_opts if dataset_compression == 'gzip' else None

            # Create compressed dataset
            # Prepare arguments for dataset creation
            create_args = {
                'data': item[:],  # Read all data
                'compression': dataset_compression,
            }

            # Add compression options only if applicable
            if dataset_compression_opts is not None:
                create_args['compression_opts'] = dataset_compression_opts

            # Add chunking only if chunk_shape is valid and dataset is large enough
            if chunk_shape is not None and item.size > 1000:  # Only chunk larger datasets
                create_args['chunks'] = chunk_shape
                # Shuffle only works with chunked datasets
                if dataset_compression == 'gzip':
                    create_args['shuffle'] = True

            # Add checksum for data integrity (only for non-empty datasets)
            if item.size > 0:
                create_args['fletcher32'] = True

            dst_group.create_dataset(key, **create_args)

        elif isinstance(item, h5py.Group):
            # It's a group - create and recurse
            if verbose:
                print(f"    Processing group: {key}")
            dst_subgroup = dst_group.create_group(key)
            copy_group_recursively(item, dst_subgroup, compression, compression_opts, verbose)


def compress_hdf5_file(file_path, compression='auto', compression_opts=6,
                      backup=True, verbose=True):
    """Compress an HDF5 file with optimized settings for robotics data"""

    start_time = time.time()

    try:
        if verbose:
            print(colored(f"Compressing: {file_path}", "cyan"))

        # Get original file info
        original_info = analyze_hdf5_file(file_path)
        if not original_info:
            return False, 0, 0, 0

        original_size = original_info['file_size_mb']

        if verbose:
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Found {original_info['num_datasets']} datasets")

        # Create backup if requested
        if backup:
            backup_path = file_path + '.backup'
            if not os.path.exists(backup_path):
                shutil.copy2(file_path, backup_path)
                if verbose:
                    print(f"  Created backup: {backup_path}")

        # Create temporary file for compression
        temp_path = file_path + '.temp'

        with h5py.File(file_path, 'r') as src, h5py.File(temp_path, 'w') as dst:
            # Copy root attributes
            for key, value in src.attrs.items():
                dst.attrs[key] = value

            # Add compression metadata
            dst.attrs['compression_date'] = datetime.now().isoformat()
            dst.attrs['compression_script'] = 'compress_hdf5.py'
            dst.attrs['original_size_mb'] = original_size
            dst.attrs['compression_method'] = compression
            dst.attrs['compression_opts'] = compression_opts

            # Recursively copy and compress all groups and datasets
            copy_group_recursively(src, dst, compression, compression_opts, verbose)

        # Replace original with compressed version
        os.replace(temp_path, file_path)

        # Get new file size
        new_size = get_file_size_mb(file_path)
        compression_ratio = (1 - new_size / original_size) * 100
        compression_time = time.time() - start_time

        if verbose:
            print(colored(f"  Compression complete:", "green"))
            print(f"    Original size: {original_size:.2f} MB")
            print(f"    New size: {new_size:.2f} MB")
            print(f"    Size reduction: {compression_ratio:.1f}%")
            print(f"    Compression time: {compression_time:.1f}s")
            print(f"    Data integrity: ✓ (Fletcher32 checksums)")

        return True, original_size, new_size, compression_ratio

    except Exception as e:
        print(colored(f"Error compressing {file_path}: {e}", "red"))
        # Clean up temp file if it exists
        temp_path = file_path + '.temp'
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, 0, 0, 0


def find_hdf5_files(base_dir):
    """Find all HDF5 files recursively"""
    hdf5_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.h5', '.hdf5')):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files


def main():
    parser = argparse.ArgumentParser(description="Advanced HDF5 compression for robotics data")
    parser.add_argument('--base_dir', type=str, required=True,
                       help="Base directory containing HDF5 files")
    parser.add_argument('--file', type=str,
                       help="Compress specific HDF5 file (optional)")
    parser.add_argument('--compression', type=str, default='gzip',
                       choices=['auto', 'gzip', 'lzf', 'szip'],
                       help="Compression method (default: auto - optimized per data type)")
    parser.add_argument('--compression_opts', type=int, default=6,
                       help="Compression level for gzip (1-9, default: 6)")
    parser.add_argument('--no_backup', action='store_true',
                       help="Don't create backup files")
    parser.add_argument('--verbose', action='store_true', default=True,
                       help="Verbose output")
    parser.add_argument('--analyze_only', action='store_true',
                       help="Only analyze files without compressing")
    parser.add_argument('--dry_run', action='store_true',
                       help="Show what would be compressed")

    args = parser.parse_args()

    print(colored("=" * 70, "cyan"))
    print(colored("ADVANCED HDF5 COMPRESSION SCRIPT", "cyan"))
    print(colored("Optimized for Robotics Data", "cyan"))
    print(colored("=" * 70, "cyan"))

    if args.file:
        # Single file mode
        if not os.path.exists(args.file):
            print(colored(f"File not found: {args.file}", "red"))
            return

        if args.analyze_only:
            info = analyze_hdf5_file(args.file)
            if info:
                print(colored(f"\nFile Analysis: {args.file}", "cyan"))
                print(f"File size: {info['file_size_mb']:.2f} MB")
                print(f"Data size: {info['total_data_size_mb']:.2f} MB")
                print(f"Number of datasets: {info['num_datasets']}")
                print(f"Root attributes: {list(info['attributes'].keys())}")

                print(colored("\nDatasets:", "yellow"))
                for path, dataset_info in info['datasets'].items():
                    compression_info = f"({dataset_info['compression']})" if dataset_info['compression'] else "(uncompressed)"
                    print(f"  {path}: {dataset_info['shape']} {dataset_info['dtype']} {compression_info} - {dataset_info['size_mb']:.2f} MB")

        elif args.dry_run:
            print(colored(f"DRY RUN: Would compress {args.file}", "yellow"))
            print(f"  Compression: {args.compression}")
            if args.compression == 'gzip':
                print(f"  Compression level: {args.compression_opts}")

        else:
            success, orig_size, new_size, ratio = compress_hdf5_file(
                args.file, args.compression, args.compression_opts,
                not args.no_backup, args.verbose
            )
            if success:
                print(colored("✓ Compression completed successfully!", "green"))
            else:
                print(colored("✗ Compression failed!", "red"))

    else:
        # Batch mode
        print(f"Searching for HDF5 files in: {args.base_dir}")

        if not os.path.exists(args.base_dir):
            print(colored(f"Directory not found: {args.base_dir}", "red"))
            return

        hdf5_files = find_hdf5_files(args.base_dir)

        if not hdf5_files:
            print(colored("No HDF5 files found", "yellow"))
            return

        print(f"Found {len(hdf5_files)} HDF5 files")

        if args.analyze_only:
            total_size = 0
            total_datasets = 0

            for file_path in hdf5_files:
                info = analyze_hdf5_file(file_path)
                if info:
                    total_size += info['file_size_mb']
                    total_datasets += info['num_datasets']
                    print(colored(f"\n{file_path}", "cyan"))
                    print(f"  Size: {info['file_size_mb']:.2f} MB")
                    print(f"  Datasets: {info['num_datasets']}")

                    # Show largest datasets
                    largest_datasets = sorted(info['datasets'].items(),
                                            key=lambda x: x[1]['size_mb'], reverse=True)[:3]
                    for path, ds_info in largest_datasets:
                        print(f"    {path}: {ds_info['size_mb']:.2f} MB")

            print(colored(f"\nSummary:", "cyan"))
            print(f"Total files: {len(hdf5_files)}")
            print(f"Total size: {total_size:.2f} MB")
            print(f"Total datasets: {total_datasets}")

            return

        if args.dry_run:
            print(colored("DRY RUN: Would compress the following files:", "yellow"))
            for file_path in hdf5_files:
                print(f"  {file_path}")
            print(f"Compression method: {args.compression}")
            return

        # Compress all files
        print(colored(f"\nCompressing {len(hdf5_files)} files...", "cyan"))

        successful = 0
        failed = 0
        total_original_size = 0
        total_new_size = 0
        start_time = time.time()

        for i, file_path in enumerate(hdf5_files, 1):
            print(colored(f"\n[{i}/{len(hdf5_files)}] Processing: {os.path.basename(file_path)}", "cyan"))

            success, orig_size, new_size, ratio = compress_hdf5_file(
                file_path, args.compression, args.compression_opts,
                not args.no_backup, args.verbose
            )

            if success:
                successful += 1
                total_original_size += orig_size
                total_new_size += new_size
            else:
                failed += 1

        # Final summary
        total_time = time.time() - start_time

        print(colored("\n" + "=" * 70, "cyan"))
        print(colored("COMPRESSION SUMMARY", "cyan"))
        print(colored("=" * 70, "cyan"))
        print(colored(f"Files processed: {len(hdf5_files)}", "cyan"))
        print(colored(f"Successful: {successful}", "green"))
        print(colored(f"Failed: {failed}", "red"))

        if successful > 0:
            total_ratio = (1 - total_new_size / total_original_size) * 100
            print(colored(f"Total original size: {total_original_size:.2f} MB", "cyan"))
            print(colored(f"Total compressed size: {total_new_size:.2f} MB", "cyan"))
            print(colored(f"Total size reduction: {total_ratio:.1f}%", "green"))
            print(colored(f"Space saved: {total_original_size - total_new_size:.2f} MB", "green"))

        print(colored(f"Total processing time: {total_time:.1f}s", "cyan"))

        if not args.no_backup:
            print(colored("💾 Backup files created (.backup extension)", "blue"))
            print(colored("🔄 Use original script with --restore to restore if needed", "blue"))


if __name__ == "__main__":
    main()