from SonataNeuralNetwork import SonataNeuralNetwork, MIDIReader
from music21.stream import Stream
from os import listdir, path
from math import log
from numpy.random import normal


music_dir = './midi/'


def stream_from_notes(notes):
    for i in xrange(100):
        inp = default + [j for i in notes for j in i][-12:]
        new = list(net.activate(inp))
        for i in new:
            new[0] += normal(dur[0], dur[1])
            new[1] += normal(freq[0], freq[1])
        notes.append(new)

    return MIDIReader.list_to_stream(notes)

if __name__ == '__main__':
    snn = SonataNeuralNetwork()
    for midi in listdir(music_dir):
        print midi
        snn.read(path.join(music_dir, midi))
    print 'training'
    net = snn.train_network()

    snn.append_errors()
    dur, freq = snn.get_error_vals()

    default = [4, 4]
    treble = stream_from_notes([[1, log(440)], [1, log(500)], [1, log(550)], [1, log(500)], [1, log(550)], [1, log(500)]])
    bass = stream_from_notes([[0.5, log(220)], [0.5, log(300)], [0.5, log(350)], [1, log(400)], [0.5, log(350)], [1, log(300)]])
    s = Stream()
    s.append(treble)
    s.append(bass)
    bass.offset = 0
    s.show()
