# HitPix Readout Python Scrips
This repository contains python classes to communicate with the hitpix readout FPGA bitstream as well as data taking and visualization scripts.

## Installation
`Python3.10` is required.
### required packages:
```
numpy matplotlib pyserial pylibftdi tqdm h5py scipy bitarray cobs rich
```
### optional packages:
```
argcomplete PyMeasure python-usbtmc python-vxi11 pyvisa
```

### Other helpful steps
- install lowlatency kernel https://packages.ubuntu.com/search?suite=all&searchon=all&keywords=lowlatency
- ubuntu python version `sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10`
- ubuntu python3 version `sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10`


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