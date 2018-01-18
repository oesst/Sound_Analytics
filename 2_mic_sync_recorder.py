import array
import struct
import time
import wave

import numpy as np
import pyaudio


### 2 Mic Recorder ###
######################


# This script simultaneously records sound from 2 input sources and stores it in two wav files
# Call it like that: python 2_mic_sync_recorder.py name_of_recorded_files recording_time


class SyncedRecorder:
    def __init__(self):
        self.pa = pyaudio.PyAudio()

        self.RATE = 44100
        self.CHUNK_SIZE = 4096
        self.FORMAT = pyaudio.paInt16

    def initMicrophones(self):
        self.stream_l = self.pa.open(format=self.FORMAT,
                                     channels=1,
                                     rate=self.RATE,
                                     input=True,
                                     output=True,
                                     input_device_index=4,
                                     frames_per_buffer=self.CHUNK_SIZE)

        self.stream_r = self.pa.open(format=self.FORMAT,
                                     channels=1,
                                     rate=self.RATE,
                                     input=True,
                                     output=True,
                                     input_device_index=5,
                                     frames_per_buffer=self.CHUNK_SIZE)

        self.output_stream = self.pa.open(format=self.pa.get_format_from_width(1),
                                          channels=1,
                                          rate=self.RATE,
                                          output=True)

    def play_sound(self):
        frequency = 500  # Hz, waves per second, 261.63=C4-note.
        duration = 5  # seconds to play sound

        number_of_frames = int(self.RATE * duration)
        rest_frames = number_of_frames % self.RATE
        wave_data = ''

        # generating wawes
        for x in range(number_of_frames):
            wave_data = wave_data + chr(int(np.sin(x / ((self.RATE / frequency) / np.pi)) * 127 + 128))

        for x in range(rest_frames):
            wave_data = wave_data + chr(128)

        self.output_stream.write(wave_data)
        self.output_stream.stop()

    def record(self, seconds):

        print("Recording %i seconds in ..." % int(recording_time))
        now = time.time()

        count = 5
        while count > 0:
            print(count)
            count -= 1
            time.sleep(1)

        now = time.time()

        r = array.array('h')
        l = array.array('h')

        # initialization
        self.initMicrophones()

        self.play_sound()



        while 1:
            # little endian, signed short
            data_l = self.stream_l.read(self.CHUNK_SIZE)
            data_r = self.stream_r.read(self.CHUNK_SIZE)

            data_l = array.array('h', data_l)
            data_r = array.array('h', data_r)

            l.extend(data_l)
            r.extend(data_r)

            if time.time() - now > seconds:
                break

        self.data_l = l
        self.data_r = r
        self.sample_size = self.pa.get_sample_size(self.FORMAT)

    def save(self, file_name):
        data_l = struct.pack('<' + ('h' * len(self.data_l)), *self.data_l)
        data_r = struct.pack('<' + ('h' * len(self.data_r)), *self.data_r)
        print("Saving recordings to files: %s_right.wav and %s_left.wav" % (file_name, file_name))

        wf = wave.open((file_name + '_left.wav'), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.sample_size)
        wf.setframerate(self.RATE)
        wf.writeframes(data_l)
        wf.close()

        wf = wave.open((file_name + '_right.wav'), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.sample_size)
        wf.setframerate(self.RATE)
        wf.writeframes(data_r)
        wf.close()

        print("DONE!")

    def close(self):
        self.stream_l.close()
        self.stream_r.close()
        self.output_stream.close()
        self.pa.terminate()


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print('Please provide exactly two arguments: recording time and file name e.g. python 2_mic_sync_recorder test 5')
        exit(1)

    recording_time = int(sys.argv[2])

    recorder = SyncedRecorder()
    recorder.record(recording_time)
    recorder.save(sys.argv[1])
    recorder.close()
    exit(0)
