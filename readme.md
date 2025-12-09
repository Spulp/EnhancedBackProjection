# Enhanced Back-Projection of Vision Features for 3D Symmetry Detection

[_WACV 2026_](https://wacv.thecvf.com/Conferences/2026) — _Work in Progress_

This repository is actively being prepared for release. Additional code modules and the associated datasets will be uploaded soon to accompany the accepted paper.

Project page: [https://spulp.github.io/EnhancedBackProjection/](https://spulp.github.io/EnhancedBackProjection/)

# Important
## Environment Setup

- **Python Version:** 3.12  
- It is recommended to use a virtual environment for package management.
- Requires GPU with CUDA 12.9.

### Creating a Virtual Environment

#### Using `venv`
```bash
python -m venv p12
source p12/bin/activate
```

#### Using Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```
```bash
conda create -n p12 python=3.12
conda install -n p12 pip
conda activate p12
```

### Installing Required Libraries

All required libraries are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```
Or install them individually:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.8.0cu129
pip install point_cloud_utils
pip install torch_geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
pip install trimesh pyvista scikit-learn
```

### References

This code is inspired by the following repositories:
- [Back-to-3D Few-Shot Keypoints](https://github.com/wimmerth/back-to-3d-few-shot-keypoints)
- [COPS](https://github.com/marco-garosi/COPS)

### Example Usage

To run the main script, use:
```bash
main.py <path_to_obj_file> <planes_or_axes> <mode>
```

- `<path_to_obj_file>`: Path to an `.obj` file (e.g., `objects/example_planes.obj`)
- `<planes_or_axes>`: Specify the planes or axes to use (e.g., `planes`)
- `<mode>`: Choose one of the following modes:
  - `FM`: Feature-Mesh sampling
  - `RM`: Raw-Mesh
  - `PC`: Point Cloud

**Example:**
```bash
main.py objects/example_planes.obj planes FM
main.py objects/example_axes.obj axes RM
```

### Notes on Code Structure and Usage

The code does not provide extensive explanations for each part, but with enough digging (e.g., looking at function names and structure), you can deduce the workflow. The entry point is the `main.py` script, which is straightforward to follow.

When executing, a visual interface pops up. The console will display commands for interacting with the viewer, summarized here:

> **Note:** There is a known bug when using the `PC` mode—the point clouds in the viewer appear larger than normal. However, the computed planes can still be visualized using the original mesh, even though the plane was computed from the point cloud.



---

**Interactive 3D Viewer Controls:**

- Use mouse to rotate/zoom the object
- Press `s` to save current camera view
- Press `r` to reset view
- Press `1` to show/hide point cloud
- Press `2` to show/hide features
- Press `3` to add plane
- Press `q` to quit

---

## Citation

If you find this useful or build on this work, please cite the paper below:

```bibtex
@InProceedings{Aguirre_2026_WACV,
  author    = {Aguirre, Isaac and Sipiran, Ivan},
  title     = {Enhanced Back-Projection of Vision Features for 3D Symmetry Detection},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
  month     = {-},
  year      = {2026},
  pages     = {-}
}
```