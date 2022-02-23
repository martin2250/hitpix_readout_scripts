import math
from typing import Literal
import numpy as np

def optimize_vco_and_divider(
    freq_in: float,
    freq_out: float,
    vco_range: tuple[float, float] = (600, 1200),
) -> tuple[float, int, float]:
    '''return feedback, output dividers and archived frequency'''
    div_fb_min = int(math.floor(vco_range[0] / freq_in))
    if div_fb_min < 1:
        div_fb_min = 1
    div_fb_max = int(math.ceil(vco_range[1] / freq_in))
    div_fb = np.arange(div_fb_min, div_fb_max + 0.1, 0.125)

    div_out_min = int(math.floor(vco_range[0] / freq_out))
    if div_out_min < 1:
        div_out_min = 1
    div_out_max = int(math.floor(vco_range[1] / freq_out))
    div_out = np.arange(div_out_min, div_out_max + 0.1)

    div_fb, div_out = np.meshgrid(div_fb, div_out)

    # find valid vco values
    freq_vco = freq_in * div_fb
    freq_vco = np.where(
        np.logical_and(freq_vco < vco_range[1], freq_vco > vco_range[0]),
        freq_vco,
        0,
    )

    # find best output values
    freq_gen = freq_vco / div_out
    freq_err = freq_gen - freq_out
    freq_err_abs = np.abs(freq_err)

    index_min = np.unravel_index(np.argmin(freq_err_abs), freq_err_abs.shape)

    return div_fb[index_min], int(div_out[index_min]), freq_gen[index_min]

def get_register_values(div_fb: float, div_serdes: int, bandwidth: Literal['low', 'low_ss', 'high', 'optimized']) -> tuple[int, int, int]:
    '''returns register values for MMCM_CONFIG_0-2
    low_ss = low spread spectrum
    div_serdes = half of serdes divider (both low and high counters)
    '''
    assert div_serdes in range(1 << 5)
    # variable names taken from xapp888/mmcme2_drp_func.h/mmcm_frac_count_calc
    # clkout0_divide_int is actually clkfb_divide_int
    clkout0_divide_int  = int(div_fb)
    clkout0_divide_frac = int(8 * (div_fb - clkout0_divide_int))
    assert clkout0_divide_int in range(64)
    assert clkout0_divide_frac in range(8)

    even_part_high = clkout0_divide_int >> 1
    even_part_low = even_part_high
    odd = clkout0_divide_int - even_part_high - even_part_low
    odd_and_frac = (8*odd) + clkout0_divide_frac

    lt_frac = even_part_high - (odd_and_frac <= 9)
    ht_frac = even_part_low  - (odd_and_frac <= 8)

    # pm_fall =  {odd[6:0],2'b00} + {6'h00, clkout0_divide_frac[2:1]}
    pm_fall =  (odd % (1 << 7)) * 4 + (clkout0_divide_frac >> 1) % (1 << 2)

    wf_fall_frac = int(((odd_and_frac >=2) and (odd_and_frac <=9)) or ((clkout0_divide_frac == 1) and (clkout0_divide_int == 2)))
    wf_rise_frac = int((odd_and_frac >=1) and (odd_and_frac <=8))

    pm_rise_frac = 0

    # pm_rise_frac_filtered = (pm_rise_frac >=8) ? (pm_rise_frac ) - 8: pm_rise_frac
    pm_rise_frac_filtered = (pm_rise_frac  - 8) if (pm_rise_frac >=8) else pm_rise_frac
    # dt_int			= dt + (& pm_rise_frac[7:4])
    pm_fall_frac		= pm_fall + pm_rise_frac
    # pm_fall_frac_filtered	= pm_fall + pm_rise_frac - {pm_fall_frac[7:3], 3'b000}
    pm_fall_frac_filtered	= pm_fall + pm_rise_frac - (pm_fall_frac & 0b11111000)

    part_mmcm_clkfb_shared = ((pm_fall_frac_filtered & 0x7) << 1) | wf_fall_frac
    part_mmcm_clkfb_reg2 = ((clkout0_divide_frac & 0x7) << 12) | (1 << 11) | (wf_rise_frac << 10)
    part_mmcm_clkfb_reg1 = ((pm_rise_frac_filtered & 0x7) << 13) | ((ht_frac & 0x3f) << 6) | (lt_frac & 0x3f)
    val_lock = _table_lock[clkout0_divide_int]
    val_filter = _tables_filter[bandwidth][clkout0_divide_int]
    
    reg0 = val_lock & ((1 << 32) - 1)
    reg1 = (part_mmcm_clkfb_reg1 << 16) | part_mmcm_clkfb_reg2
    reg2 = (part_mmcm_clkfb_shared << 23) | (((val_lock >> 32) & 0xff) << 15) | (div_serdes << 10) | val_filter

    return reg0, reg1, reg2

################################################################################
# lookup tables from xapp888

_table_lock = [
    # index: divider
    # LockRefDly_LockFBDly_LockCnt_LockSatHigh_UnlockCnt
    0b00110_00110_1111101000_1111101001_0000000001,
    0b00110_00110_1111101000_1111101001_0000000001,
    0b01000_01000_1111101000_1111101001_0000000001,
    0b01011_01011_1111101000_1111101001_0000000001,
    0b01110_01110_1111101000_1111101001_0000000001,
    0b10001_10001_1111101000_1111101001_0000000001,
    0b10011_10011_1111101000_1111101001_0000000001,
    0b10110_10110_1111101000_1111101001_0000000001,
    0b11001_11001_1111101000_1111101001_0000000001,
    0b11100_11100_1111101000_1111101001_0000000001,
    0b11111_11111_1110000100_1111101001_0000000001,
    0b11111_11111_1100111001_1111101001_0000000001,
    0b11111_11111_1011101110_1111101001_0000000001,
    0b11111_11111_1010111100_1111101001_0000000001,
    0b11111_11111_1010001010_1111101001_0000000001,
    0b11111_11111_1001110001_1111101001_0000000001,
    0b11111_11111_1000111111_1111101001_0000000001,
    0b11111_11111_1000100110_1111101001_0000000001,
    0b11111_11111_1000001101_1111101001_0000000001,
    0b11111_11111_0111110100_1111101001_0000000001,
    0b11111_11111_0111011011_1111101001_0000000001,
    0b11111_11111_0111000010_1111101001_0000000001,
    0b11111_11111_0110101001_1111101001_0000000001,
    0b11111_11111_0110010000_1111101001_0000000001,
    0b11111_11111_0110010000_1111101001_0000000001,
    0b11111_11111_0101110111_1111101001_0000000001,
    0b11111_11111_0101011110_1111101001_0000000001,
    0b11111_11111_0101011110_1111101001_0000000001,
    0b11111_11111_0101000101_1111101001_0000000001,
    0b11111_11111_0101000101_1111101001_0000000001,
    0b11111_11111_0100101100_1111101001_0000000001,
    0b11111_11111_0100101100_1111101001_0000000001,
    0b11111_11111_0100101100_1111101001_0000000001,
    0b11111_11111_0100010011_1111101001_0000000001,
    0b11111_11111_0100010011_1111101001_0000000001,
    0b11111_11111_0100010011_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
    0b11111_11111_0011111010_1111101001_0000000001,
]

_table_filter_low = [
    # filter settings for low PLL bandwidth
    # index: divider
    # CP_RES_LFHF
    0b0010_1111_00,
    0b0010_1111_00,
    0b0010_1111_00,
    0b0010_1111_00,
    0b0010_0111_00,
    0b0010_1011_00,
    0b0010_1101_00,
    0b0010_0011_00,
    0b0010_0101_00,
    0b0010_0101_00,
    0b0010_1001_00,
    0b0010_1110_00,
    0b0010_1110_00,
    0b0010_1110_00,
    0b0010_1110_00,
    0b0010_0001_00,
    0b0010_0001_00,
    0b0010_0001_00,
    0b0010_0110_00,
    0b0010_0110_00,
    0b0010_0110_00,
    0b0010_0110_00,
    0b0010_0110_00,
    0b0010_0110_00,
    0b0010_0110_00,
    0b0010_1010_00,
    0b0010_1010_00,
    0b0010_1010_00,
    0b0010_1010_00,
    0b0010_1010_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_1100_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
    0b0010_0010_00,
]

_table_filter_low_ss = [
    # filter settings for low PLL bandwidth
    # index: divider
    # CP_RES_LFHF
    0b0010_1111_11,
    0b0010_1111_11,
    0b0010_1111_11,
    0b0010_1111_11,
    0b0010_0111_11,
    0b0010_1011_11,
    0b0010_1101_11,
    0b0010_0011_11,
    0b0010_0101_11,
    0b0010_0101_11,
    0b0010_1001_11,
    0b0010_1110_11,
    0b0010_1110_11,
    0b0010_1110_11,
    0b0010_1110_11,
    0b0010_0001_11,
    0b0010_0001_11,
    0b0010_0001_11,
    0b0010_0110_11,
    0b0010_0110_11,
    0b0010_0110_11,
    0b0010_0110_11,
    0b0010_0110_11,
    0b0010_0110_11,
    0b0010_0110_11,
    0b0010_1010_11,
    0b0010_1010_11,
    0b0010_1010_11,
    0b0010_1010_11,
    0b0010_1010_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_1100_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
    0b0010_0010_11,
]

_table_filter_high = [
    # filter settings for low PLL bandwidth
    # index: divider
    # CP_RES_LFHF
    0b0010_1111_00,
    0b0100_1111_00,
    0b0101_1011_00,
    0b0111_0111_00,
    0b1101_0111_00,
    0b1110_1011_00,
    0b1110_1101_00,
    0b1111_0011_00,
    0b1110_0101_00,
    0b1111_0101_00,
    0b1111_1001_00,
    0b1101_0001_00,
    0b1111_1001_00,
    0b1111_1001_00,
    0b1111_1001_00,
    0b1111_1001_00,
    0b1111_0101_00,
    0b1111_0101_00,
    0b1100_0001_00,
    0b1100_0001_00,
    0b1100_0001_00,
    0b0101_1100_00,
    0b0101_1100_00,
    0b0101_1100_00,
    0b0101_1100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0010_1000_00,
    0b0010_1000_00,
    0b0010_1000_00,
    0b0010_1000_00,
    0b0010_1000_00,
    0b0111_0001_00,
    0b0111_0001_00,
    0b0100_1100_00,
    0b0100_1100_00,
    0b0100_1100_00,
    0b0100_1100_00,
    0b0110_0001_00,
    0b0110_0001_00,
    0b0101_0110_00,
    0b0101_0110_00,
    0b0101_0110_00,
    0b0010_0100_00,
    0b0010_0100_00,
    0b0010_0100_00,
    0b0010_0100_00,
    0b0100_1010_00,
    0b0011_1100_00,
    0b0011_1100_00,
]

_table_filter_optimized = [
    # filter settings for low PLL bandwidth
    # index: divider
    # CP_RES_LFHF
    0b0010_1111_00,
    0b0100_1111_00,
    0b0101_1011_00,
    0b0111_0111_00,
    0b1101_0111_00,
    0b1110_1011_00,
    0b1110_1101_00,
    0b1111_0011_00,
    0b1110_0101_00,
    0b1111_0101_00,
    0b1111_1001_00,
    0b1101_0001_00,
    0b1111_1001_00,
    0b1111_1001_00,
    0b1111_1001_00,
    0b1111_1001_00,
    0b1111_0101_00,
    0b1111_0101_00,
    0b1100_0001_00,
    0b1100_0001_00,
    0b1100_0001_00,
    0b0101_1100_00,
    0b0101_1100_00,
    0b0101_1100_00,
    0b0101_1100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0011_0100_00,
    0b0010_1000_00,
    0b0010_1000_00,
    0b0010_1000_00,
    0b0010_1000_00,
    0b0010_1000_00,
    0b0111_0001_00,
    0b0111_0001_00,
    0b0100_1100_00,
    0b0100_1100_00,
    0b0100_1100_00,
    0b0100_1100_00,
    0b0110_0001_00,
    0b0110_0001_00,
    0b0101_0110_00,
    0b0101_0110_00,
    0b0101_0110_00,
    0b0010_0100_00,
    0b0010_0100_00,
    0b0010_0100_00,
    0b0010_0100_00,
    0b0100_1010_00,
    0b0011_1100_00,
    0b0011_1100_00,
]

_tables_filter = {
    'low': _table_filter_low,
    'low_ss': _table_filter_low_ss,
    'high': _table_filter_high,
    'optimized': _table_filter_optimized,
}
