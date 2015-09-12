from SonataNeuralNetwork import SonataNeuralNetwork, MIDIReader
from music21.stream import Stream
from os import listdir, path


music_dir = '/Users/phrayezzen/Documents/Projects/hackathons/SonataInPython/midi/'


def stream_from_notes(notes):
    for i in xrange(100):
        inp = default + notes[-4] + notes[-3] + notes[-2] + notes[-1]
        notes.append(list(net.activate(inp)))

    return MIDIReader.list_to_stream(notes)

if __name__ == '__main__':
    snn = SonataNeuralNetwork()
    for midi in listdir('./midi'):
        print midi
        snn.read(path.join(music_dir, midi))
    print 'training'
    net = snn.train_network()

    # b_net = bassNN.train_network()

    default = [4, 4]
    treble = stream_from_notes([[1, 440], [1, 500], [1, 550], [1, 500]])
    bass = stream_from_notes([[0.5, 220], [0.5, 300], [0.5, 350], [1, 400], [0.5, 350], [0.5, 300]])
    s = Stream()
    s.append(treble)
    s.append(bass)
    bass.offset = 0
    s.show()
