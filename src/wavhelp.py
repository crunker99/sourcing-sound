import struct

class WavHelper():

    def get_file_props(self, fname):

        wavfile = open(fname, 'rb')

        riff = wavfile.read(12)
        fmt = wavfile.read(36)

        num_channels_str = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_str)[0]

        sample_rate_str = fmt[12:16]
        sample_rate = struct.unpack('<I', sample_rate_str)[0]

        bit_depth_str = fmt[22:24]
        bit_depth = struct.unpack('<H', bit_depth_str)[0]

        return num_channels, sample_rate, bit_depth