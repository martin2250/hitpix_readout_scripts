#pylint:disable = C0302
"""The Lecroy module

All LeCroy devices are implemented in this module.
"""

import struct
import datetime
import logging
from typing import Any, Union
import usbtmc
import vxi11

DATAFORMATS = {1:"b", 2:"h", 4:"i"}

class Connection:
    def write(self, command: str) -> None:
        raise NotImplementedError()

    def read(self, command=None, decode="utf-8") -> Any:
        raise NotImplementedError()

class TMCConnection(Connection):
    def __init__(self, vendor: int = 0x05ff, device: int = 0x1023):
        self._port = usbtmc.Instrument(vendor, device)

    def write(self, command: str) -> None:
        self._port.write(command)

    def read(self, command=None, decode="utf-8") -> Any:
        if command is not None:
            self.write(command)
        if decode is None:
            return self._port.read_raw()
        else:
            return self._port.read()

class VXI11Connection(Connection):
    def __init__(self, ip: str) -> None:
        self._port = vxi11.Instrument(ip)
    
    def write(self, command: str) -> None:
        self._port.write(command)

    def read(self, command=None, decode="utf-8") -> Any:
        if command is not None:
            self.write(command)
        if decode is None:
            return self._port.read_raw()
        else:
            return self._port.read()

class Lecroy(): #pylint: disable=R0904
    """ The Lecroy baseclass. It contains methods to communicate with Lecroy
    oscilloscopes.
    """
    def __init__(self, connection: Connection):
        """ init function for the iseq. this method will not be necessary for
            the corrected device class.
        """
        self.connection = connection

        self._last_data = None
        self.log = logging.getLogger(__name__)
    
    def write(self, command: str) -> None:
        self.connection.write(command)

    def read(self, command=None, decode="utf-8") -> Any:
        return self.connection.read(command, decode)

    def init(self):
        """Initialize the oscilloscope

        The initialization routine consists of following steps:
        if the oscilloscope is switched on nothing more to do

        """
        self._last_data = None
        self.log.info("Initialize the scope")
        #reset the scope
        self.reset()
        self.reset()
        self.set_response_format("SHORT")
        # disable all channels
        for i in range(4):
            self.disable_channel(i+1)
        self.log.info(self.identify())

    def ramp_down(self):
        """ Ramp down method for the scopes, resets the scope"""


    def set_response_format(self, commformat="SHORT"):
        """
        set the response format

        Args:
          commformat (string)
            response format of the scope (SHORT, LONG, OFF)
            OFF is not recommended, since it is not provided for
            the query commands!
        """
        if commformat == "OFF":
            self.log.warning("this format is not provided!")
        self.write("CHDR {0}".format(commformat))

    def identify(self):
        """ Ask for the identifier of the scope"""
        return self.read("*IDN?")

    def reset(self):
        """ send a reset command to the scope"""
        self.write("*RST")

    def set_trigger_source(self, trigger="C2", trigger_type="EDGE"):
        """ set the trigger source
        Args:
            trigger (string) - trigger source (C1-C4,EX,LINE,EX10,PA,ETM10)
            trigger_type (string) - trigger type:
                                    EDGE, DROP, GLIT, INTV, STD, SNG, SQ, TEQ
        """
        self.write("TRSE {0}, SR, {1}".format(trigger_type, trigger))

    def get_trigger_source(self):
        """ get the trigger source
        Returns:
            source (string) - trigger source
        """
        answer = self.read("TRSE?").split(",")[2]
        return answer

    def get_enabled_channels(self):
        """ get enabled channels
        Returns:
            enabled_channels (list) : list of enabled channels
        """
        enabled_channels = []
        for channel in range(4):
            answer = self.read("C{0}:TRA?".format(channel+1))
            if "ON" in answer:
                enabled_channels.append(channel+1)
        return enabled_channels

    def enable_channel(self, channel):
        """ enable the channel
        Args:
            channel (int) - channel to enable
        """
        self.write("C{0}:TRA ON".format(channel))

    def disable_channel(self, channel):
        """ disable the channel
        Args:
            channel (int) - channel to disable
        """
        self.write("C{0}:TRA OFF".format(channel))

    def combine_channels(self, combine=True):
        """ combine channels for higher acquisition rate 20GS to 40Gs
        """
        self.write("COMB {0}".format(combine+1))

    def set_trigger_level(self, level, source=None):
        """ set the trigger level and the source
        Args:
            level (float) - trigger level in volts
        Kwargs:
            source (string) - trigger source
        """
        if source is None:
            source = self.get_trigger_source()
        self.write("{0}:TRLV {1}V".format(source, level))

    def get_trigger_level(self, source=None):
        """
        set the trigger level and the source

        Kwargs:
          source (string) - trigger source
        """
        if source is None:
            source = self.get_trigger_source()
        answer = self.read("{0}:TRLV?".format(source)).replace("V", "")
        return float(answer.split(" ")[1])

    def set_voltage_offset(self, offset, channel):
        """Set the channel offset in y direction (Voltage)
        Args:
            offset (float)     -   y offset in Volts
        """
        self.write("C{0}:OFFSET {1}".format(channel, offset))

    def get_voltage_offset(self, channel):
        """Get the channel offset in y direction (Voltage)
        Returns:
            Offset (float)      - y offset in Volts
        """
        offset = self.read("C{0}:OFFSET?".format(channel))
        return float(offset.split(" ")[1])

    def auto_voltage_range(self, channel=None, channels=None):
        """ adjust automatically the voltage range of a channel. If no channel
        or channels are defined for all enabled channels the autorange is
        performed for all enabled channels
        Args:
            channel (int)   - channel to apply the auto voltage range
            channels (list) - list of channels for which the voltage range
                              will be adjusted
        """
        chans = []
        if channels is not None:
            chans = channels
        if channel is not None:
            chans.append(channel)
        if chans == []:
            chans = self.get_enabled_channels()
        for chan in chans:
            print("adjust channel ", chan)
            self.write("C{0}:ASET FIND".format(chan))

    def set_trigger_delay(self, delay):
        """ set the delay of the trigger
        Args:
            delay (float) - trigger delay in seconds
        """
        self.write("TRDL {0}S".format(delay))

    def get_trigger_delay(self):
        """ get the delay of the trigger
        returns:
            delay (float) - trigger delay in seconds
        """
        delay = self.read("TRDL?")
        return float(delay.split(" ")[1])

    def set_termination(self, termination, channel):
        """
        Set the channel termination in Ohm (possible values 50Ohm or 1MOhm)

        Args:
          termination (string)
            Channel termination and coupling
              A1M, D1M -> 1MOhm with AC or DC coupling
              D50      -> 50Ohm with DC coupling
              GND      -> Ground
          channel (string)
            of which the coupling has to be set
        """
        self.write("{0}:CPL {1}".format(channel, termination))

    def get_termination(self, channel):
        """Get the Termination of the measured channel in Ohm
        Args:
            channel (string) - defines the channel of which the termination
                               has to be returned
        Returns:
            Termination (int)   -   returns the Termination in Ohm
        """
        termination = self.read("{0}:CPL?".format(channel))
        return termination.split(" ")[1]

    def set_time_base(self, timebase):
        """ Set the time base of the scope in seconds per division
        Args:
            timebase (float) - timebase in seconds per division
        """
        self.write("TDIV {0}S".format(timebase))

    def get_time_base(self):
        """ Set the time base of the scope in seconds per division
        returns:
            timebase (float) - timebase in seconds per division
        """
        timebase = self.read("TDIV?")
        return float(timebase.split(" ")[1])

    def set_time_range(self, timerange):
        """Set the range in horizontal direction in seconds (x-Axis)
        Args:
            timerange  (float) -  Set the time range of the scope measurement
                                  in seconds
        """
        self.set_time_base(float(timerange)/10.0)

    def get_time_range(self):
        """Get the horizontal time range of the scope in seconds

        Returns:
            timerange (float)   -   returns the time range in x direction in
                                    seconds
        """
        return self.get_time_base()*10.0

    def set_voltage_base(self, voltagebase, channel):
        """ Set the voltage base of a given channel in volts per division
        Args:
            voltagebase (float) - voltage base in voltage per division
            channel     (int)   - channel of which the voltage base is changed
        """
        self.write("C{0}:VDIV {1}V".format(channel, voltagebase))

    def get_voltage_base(self, channel):
        """ returns the voltage base of a given channel in volts per divisions
        Args:
            channel (int)   - channel of which the voltage range is read
        Returns:
            voltagebase (float) - voltage base of the given channel in volts
                                  per division
        """
        voltagebase = self.read("C{0}:VDIV?".format(channel))
        return float(voltagebase.split(" ")[1])

    def set_voltage_range(self, voltagerange, channel):
        """Set the voltage range in Volts

        Args:
            voltagerange (float)  -   voltage in volts
            channel (int)         -   channel to apply the range
        """
        voltagebase = float(voltagerange)/8.0
        self.set_voltage_base(voltagebase, channel)

    def get_voltage_range(self, channel):
        """Get the voltage range in Volts
        Args:
            channel (int)       - channel for which the voltage range is
                                  returned
        Returns:
            voltagerange (float)  -   voltage range in volts
        """
        return self.get_voltage_base(channel)*8.0

    def set_trigger_mode(self, mode):
        """ Set the mode of the trigger: AUTO, NORM, SINGLE, STOP
        Args:
            mode (string) - trigger mode
        """
        self.write("TRMD {0}".format(mode))

    def get_trigger_mode(self):
        """ Get the mode of the trigger: AUTO, NORM, SINGLE, STOP
        Returns:
            mode (string) - trigger mode
        """
        return self.read("TRMD?").split(" ")[1]

    def set_trigger_slope(self, slope, trigger_source=None):
        """Set the Trigger mode (Rising or Falling Edge)
        Args:
            slope(string)   -   "NEG" or "POS" for negative or positive slope
        Kwargs:
            trigger_source (string) - Trigger source: Cx, LINE, EX, EX10, ETM10
                                      if not defined the current trigger source
                                      is used
        """
        if trigger_source is None:
            trigger_source = self.get_trigger_source()
        self.write("{0}:TRSL {1}".format(trigger_source, slope))

    def get_trigger_slope(self, trigger_source=None):
        """
        Get the Trigger Mode (Rising or falling edge)

        Kwargs:
          trigger_source (string):
            trigger source of which the mode is returned,
            by default the currently used trigger

        Returns:
            trigger_slope (String)    -   POS or NEG edge
        """
        if trigger_source is None:
            trigger_source = self.get_trigger_source()
        slope = self.read("{0}:TRSL?".format(trigger_source))
        return slope.split(" ")[1]

    def get_pulses(self, numpulses, channels=None, **kwargs):
        """ Get a number of valid pulses. This is for speeding up the TCT
        measurements
        Returns:
            list of valid pulses
        """
        pulselist = []
        for pulse in range(numpulses): #pylint: disable=W0612
            pulselist.append(self.get_pulse(channels, **kwargs))
        return pulselist

    def get_pulse(self, channels=None, **kwargs):# pylint: disable=R0914,W0613
        """Method to call to record one data pulse (trigger event).
        Then from the data voltage over time is calculated

        Returns:
            Data (list) -   list comprising two entries:
                             1. "time": time points
                             2. "voltage": related voltage points
        args:
            channels (list of int) - list of channels to read out, by default
                                     read all enabled channels.
        """
        if channels is None:
            channels = self.get_enabled_channels()
        self.write("TRMD STOP; ARM_ACQUISITION;TRIG_MODE SINGLE; *WAI")
        self.read("*OPC?")
        result = {}
        time_stamp = str(datetime.datetime.now())
        for channel in channels:
            result[channel] = {}
            self.write("C{0}:WAVEFORM?".format(channel))
            data = self.read(decode=None)
            while data == self._last_data:
                self.log.error("Same data, read again")
                self.write("C{0}:WAVEFORM?".format(channel))
                data = self.read(decode=None)
            self._last_data = data
            data = data[data.find(b"WAVEDESC"):]
            #Get Data Format from the waveform
            endian = data[34]
            if endian == 1: #low byte first (big endian)
                endian = "<"
            if endian == 0: #high byte first (little endian)
                endian = ">"
            descriptorlength = struct.unpack(endian+"I", data[36:40])[0]
            descriptor = data[:descriptorlength]
            dat_length = struct.unpack(endian+"I", descriptor[60:64])[0]
            dat_count = struct.unpack(endian+"I", descriptor[116:120])[0]
            point_data = data[descriptorlength:descriptorlength+dat_length]
            #unpack the data
            fmt = endian + dat_count*\
                           DATAFORMATS[dat_length/dat_count]
            y_pointdata = struct.unpack(fmt, point_data)
            y_gain = struct.unpack(endian+"f", descriptor[156:160])[0]
            y_offset = struct.unpack(endian+"f", descriptor[160:164])[0]
            x_interval = struct.unpack(endian+"f", descriptor[176:180])[0]
            x_offset = struct.unpack(endian+"d", descriptor[180:188])[0]
            y_volts = []
            x_times = []
            for pos, point in enumerate(y_pointdata):
                y_volts.append(point*y_gain - y_offset)
                x_times.append(pos*x_interval + x_offset)
            result[channel]["time"] = x_times
            result[channel]["voltage"] = y_volts
            result[channel]["time_stamp"] = time_stamp
        return result

import numpy as np

class Waverunner8404M(Lecroy): #pylint: disable=R0904, R0902
    """The Waverunner 8404M oscilloscope

    4 Channel Oscilloscope with 40GS/s(20GS/s) and 4GHz Bandwidth
    It is based on the Lecroy baseclass and adds specific methods for the
    Waverunner 8404M scope
    WARNING:
    - problems with usbtmc when connecting scope with USB2 cable to USB3 port!
    """

    def identifier(self):
        """ Check wether the scope has the correct identifier"""
        return "WAVERUNNER8404M" in self.read("*IDN?")

    def get_sequence_pulses(self, numpulses, channels=None,
                            max_datalength="20000"):
        """record a sequence of pulses and read it out
        Args:
            numpulses (int) - number of pulses to read out
            max_datalength (string) - see record_sequence()
        Returns:
            pulselist (list) - list of dictionaries containing the pulse
                               information
        """
        success = self.record_sequence(numpulses, max_datalength)
        if success:
            return self.get_sequence_data(channels)
        self.log.error("No data recorded!")
        return []

    def record_sequence(self, num_triggers, max_datalength="20000", wait=False):
        """
        record a sequence of num_triggers trigger events

        Args:
          num_triggers (int)
            number of triggers to be recorded in the sequence
          max_datalength (string)
            maximum data length: e.g. number of points to be recorded or
            size of one sequence like 1M for one Megabyte
          wait (bool)
            Wait for data to be ready, otherwise only wait for
            single shot to start

        Returns:
          success (bool)
            True if successfully recorded otherwise False
        """
        success = False
        try:
            self.write("TRMD STOP")
            self.write("SEQ ON, {0}, {1}".format(num_triggers, max_datalength))
            self.write("TRMD SINGLE")
            if wait:
                self.write("WAIT")
            success = bool(self.read("*OPC?").split(" ")[1])
        except: #pylint: disable=W0702
            self.log.error("too many sequences recorded. Ran in timeout!")
        return success
    
    def wait_complete(self) -> bool:
        self.write("WAIT")
        return bool(self.read("*OPC?").split(" ")[1])

    def get_sequence_data(self, channels=None) -> tuple[np.ndarray, np.ndarray, float, float]: #pylint: disable=R0914
        """ get the data of the sequence which is currently recorded
        Args:
            channels (list of int) - list of channels to readout by default the
                                     enabled channels are used
        
        Returns:
            y_data[channel][event][point] : np.ndarray
            trigger_times[event]: np.ndarray
            time_offset: float
            time_delta: float
        """
        if channels is None:
            channels = self.get_enabled_channels()
        # read binary data
        binary_data = []
        descriptors = []
        for channel in channels:
            self.write("C{0}:WF?".format(channel))
            data = self.read(decode=None)
            descriptors.append(Descriptor(data))
            binary_data.append(data[data.find(b"WAVEDESC"):])
        
        num_triggers = descriptors[0].num_triggers
        data_length = descriptors[0].data_length
        data_count = descriptors[0].data_count
        endian = descriptors[0].endian
        # data stored as 8/16/32 bit values?
        point_format = endian + DATAFORMATS[data_length/data_count]
        # trigger offsets (double values)
        # return only every second value
        trigger_times = np.frombuffer(
            buffer=binary_data[0],
            dtype = f'{endian}d',
            count=(descriptors[0].len_trigtimes // 8),
            offset=descriptors[0].length,
        )[::2]

        y_data = np.zeros((
            len(channels),
            num_triggers,
            data_count // num_triggers,
        ))

        for i_channel, (data, desc) in enumerate(zip(binary_data, descriptors)):
            desc: Descriptor = desc
            y_data_chan = np.frombuffer(
                buffer=data,
                dtype=point_format,
                count=data_count,
                offset=desc.length + desc.len_trigtimes,
            )
            y_data_rescaled = y_data_chan * desc.voltage_gain - desc.voltage_offset
            y_data[i_channel] = np.reshape(y_data_rescaled, (num_triggers, -1))

        delta_time = descriptors[0].delta_time
        time_offset = descriptors[0].time_offset

        return y_data, trigger_times, time_offset, delta_time

class Descriptor: #pylint: disable=R0902, R0903
    """ Descriptor class to decript the descriptor of the scope data
    """
    def __init__(self, data=None):
        """ Constructor of the descriptor class
        Args:
            data (binary string) - binary Waveform recorded from the scope
        """
        self.endian = None
        self.num_triggers = None
        self.length: int = None
        self.len_trigtimes: int = None
        self.data_length = None # number of bytes
        self.data_count = None # number of values in those bytes
        self.voltage_gain = None
        self.voltage_offset = None
        self.delta_time = None
        self.time_offset = None
        self.instrument_name = None
        if data is not None:
            self.load_data(data)

    def load_data(self, data):
        """ Load the data in the internal attributes of the descriptor
        Args:
            data (binary string) - binary Waveform recorded from the scope
        """
        data = data[data.find(b"WAVEDESC"):]
        endian = data[34]
        if endian == 1: #low byte first (big endian)
            endian = "<"
        if endian == 0: #high byte first (little endian)
            endian = ">"
        self.endian = endian
        self.num_triggers = struct.unpack(endian + "H", data[174:176])[0]
        self.length = struct.unpack(endian+"I", data[36:40])[0]
        self.len_trigtimes = struct.unpack(endian+"I", data[48:52])[0]
        self.data_length = struct.unpack(endian+"I", data[60:64])[0]
        self.data_count = struct.unpack(endian+"I", data[116:120])[0]
        self.voltage_gain = struct.unpack(endian+"f", data[156:160])[0]
        self.voltage_offset = struct.unpack(endian+"f", data[160:164])[0]
        self.delta_time = struct.unpack(endian+"f", data[176:180])[0]
        self.time_offset = struct.unpack(endian+"d", data[180:188])[0]
        self.instrument_name = data[76:92].decode("utf-8")
