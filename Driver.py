from SonataNeuralNetwork import SonataNeuralNetwork, MIDIReader
from music21.stream import Score
from music21 import clef
from os import listdir, path
from math import log, e
from random import random


music_dir = './slow/'


def new_freq(old_freq, dist):
    return log(2 ** (dist / 12) * e ** old_freq)


def stream_from_notes(notes, net, dur, freq):
    total_dur = 0
    while total_dur < 100:
        inp = [j for i in notes for j in i][-12:] + [notes[-3][1], notes[-2][1], notes[-1][1], notes[-1][1]]
        new = list(net.activate(inp))

        r = random()
        if r < 0.05:
            new[1] = new_freq(new[1], 5)
        elif r < 0.1:
            new[1] = new_freq(new[1], 7)
        elif r < 0.15:
            new[1] = new_freq(new[1], 12)
        elif r < 0.2:
            new[1] = new_freq(new[1], -5)
        elif r < 0.25:
            new[1] = new_freq(new[1], -7)
        elif r < 0.3:
            new[1] = new_freq(new[1], -12)

        # r = random()
        # if r < 0.01:
        #     new[0] -= 0.25
        # elif r < 0.02:
        #     new[0] += 0.25
        # elif r < 0.03:
        #     new[0] += 0.5
        # elif r < 0.04:
        #     new[0] += 1
        # elif r < 0.05:
        #     new[0] += 1.5
        # elif r < 0.06:
        #     new[0] += 2
        # for i in new:
        #     new[0] += normal(dur[0], dur[1])
        #     new[1] += normal(freq[0], freq[1])
        notes.append(new)

        total_dur += new[0]

    return MIDIReader.list_to_stream(notes)

if __name__ == '__main__':
    snn = SonataNeuralNetwork()
    for midi in listdir(music_dir):
        print midi
        snn.read(path.join(music_dir, midi))
    print 'training'
    t_net, b_net = snn.train_network()

    snn.append_errors()
    td, tf, bd, bf = snn.get_error_vals()

    treble = stream_from_notes([[1, log(261.6)], [1, log(329.6)], [1, log(392)], [1, log(329.6)], [1, log(392)], [1, log(523.3)]],
                               t_net, td, tf)
    treble.insert(0, clef.TrebleClef())
    bass = stream_from_notes([[0.5, log(130.8)], [0.5, log(164.8)], [0.5, log(196)], [1, log(261.6)], [0.5, log(196)], [1, log(164.8)]],
                             b_net, bd, bf)
    bass.insert(0, clef.BassClef())
    s = Score()
    s.append(treble)
    s.append(bass)
    bass.offset = 0
    s.show()
