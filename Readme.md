# HitPix Readout Python Scrips
This repository contains python classes to communicate with the hitpix readout FPGA bitstream as well as data taking and visualization scripts.

## Installation
`Python3.10` is required.
### required packages:
```
numpy matplotlib pyserial pylibftdi tqdm h5py scipy bitarray cobs
```
### optional packages:
```
argcomplete PyMeasure python-usbtmc python-vxi11 pyvisa rich
```



#### DAC settings for ampout
```python
dac_default_hitpix2 = {
    'enable_output_cmos': True,
    'enable_output_diff': True,
    'enable_bandgap': False,
    'unlock': 0b101,
    'iblres': 1,
    'vn1': 30,
    'infb': 1,
    'vnfoll': 1,
    'vndel': 8,
    'vn2': 0,
    'infb2': 4,
    'ipload2': 3,
    'vncomp': 1,
    'ipfoll': 16,
    'vth': 195,
}
```