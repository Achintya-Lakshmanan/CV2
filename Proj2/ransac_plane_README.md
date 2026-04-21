# RANSAC 3D Plane Fitting Documentation

## Overview
This script implements the RANSAC algorithm to find the largest subset of 3D points that fit a plane equation of the form $ax + by + cz + d = 0$.

## Key Steps
1. **Minimum Points**: 3 points are needed to define a plane in 3D.
2. **Random Sampling**: In each iteration, 3 unique points are randomly sampled.
3. **Plane Fitting**: The plane equation is computed using the cross product of vectors formed by the 3 points.
4. **Inlier Detection**: For each candidate plane, all points are checked for their perpendicular distance to the plane. Points within a threshold are counted as inliers.
5. **Best Model Selection**: The plane with the largest set of inliers after all iterations is selected as the dominant plane.

## Usage
- The main function demonstrates usage with synthetic data.
- Adjust `threshold` and `max_iterations` for your dataset.

## Restrictions
- No use of external RANSAC libraries (OpenCV, Matlab, etc.).
- Pure NumPy implementation.

## Example
Run the script directly to see an example with random points and outliers.

---

For questions, see the code comments or ask for further clarification.
