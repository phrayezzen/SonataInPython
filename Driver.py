from SonataNeuralNetwork import SonataNeuralNetwork, MIDIReader
from music21.stream import Stream
from music21 import clef
from os import listdir, path
from math import log
from random import random


music_dir = './newmidi/'


def new_freq(old_freq, dist):
    return 2 ** (dist / 12) * old_freq


def stream_from_notes(notes, net, dur, freq):
    for i in xrange(100):
        inp = default + [j for i in notes for j in i][-12:] + [notes[-3][1], notes[-2][1], notes[-1][1], notes[-1][1]]
        new = list(net.activate(inp))

        r = random()
        if r < 0.2:
            new[1] = new_freq(new[1], 5)
        elif r < 0.4:
            new[1] = new_freq(new[1], 7)
        elif r < 0.6:
            new[1] = new_freq(new[1], 12)
        # for i in new:
        #     new[0] += normal(dur[0], dur[1])
        #     new[1] += normal(freq[0], freq[1])
        notes.append(new)

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

    default = [4, 4]
    treble = stream_from_notes([[1, log(440)], [1, log(500)], [1, log(550)], [1, log(500)], [1, log(550)], [1, log(500)]],
                               t_net, td, tf)
    treble.append(0, clef.TrebleClef())
    bass = stream_from_notes([[0.5, log(220)], [0.5, log(300)], [0.5, log(350)], [1, log(400)], [0.5, log(350)], [1, log(300)]],
                             b_net, bd, bf)
    bass.insert(0, clef.BassClef())
    s = Stream()
    s.append(treble)
    s.append(bass)
    bass.offset = 0
    s.show()
