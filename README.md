# pyoffaxis

A Python library for complex field retrieval using off-axis holography.

## Install

```bash
pip install git+https://github.com/wavefrontshaping/pyoffaxis.git
```

Optionally, install with `Cuda` GPU support (using `cupy`) with

```bash
pip install git+https://github.com/wavefrontshaping/pyoffaxis.git .[gpu]
```

## Usage

### Initialize the `Holography` object:

`raw_ref_img` is the image of the reference intensity **only**.
It should be the same size as the interference frames to process.


```python
holo2 = pyoffaxis.Holography(
    dim = shape,
    reference=raw_ref_img,
    display = True,
    padding = 100,
    sigma_noise = sigma_noise,
    use_gpu=False
)
```


### Calibrate with a stack of interferences images


```python
holo2.calibrate(frames,
               threshold_coeff = 0.5,
               radius_mask_coeff = 0.225,       # for mask around DC
               axis_mask_width = 4,             # mask around the horizontal and vertical axis
               mask_ratio = 0.75, 
               )
```

### Process a stack of frames

Using `FFT`:

```python
complex_fields = holo2.getFieldStack(frames, do_filter_ref=False)
```

Using `FFTzoom`:

```python
complex_fields = holo2.getFieldStackZoom(frames, do_filter_ref=False)
```