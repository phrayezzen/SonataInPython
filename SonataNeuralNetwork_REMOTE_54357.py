from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import music21 as m2


class MIDIReader(object):

    @staticmethod
    def parse(filename):
        piece = {}
        piece['score'] = m2.converter.parse(filename)
        piece['key_sigs'] = [(sig.sharps, sig.getOffsetBySite(piece['score'][0]))
                             for sig in list(piece['score'][0].getElementsByClass(m2.key.KeySignature))]
        piece['time_sigs'] = [(sig.beat, sig.denominator, sig.getOffsetBySite(piece['score'][0]))
                              for sig in list(piece['score'][0].getElementsByClass(m2.meter.TimeSignature))]
        return piece

    @staticmethod
    def get_notes(staff, key_sig):
        key_sig_adjust = {-7: 1, -6: 6, -5: -1, -4: 4, -3: -3, -2: 2, -1: -5, 0: 0, 1: 5, 2: -2, 3: 3, 4: -4, 5: 1, 6: -6, 7: -1}
        return [MIDIReader.get_chord_or_note(note).transpose(key_sig_adjust[key_sig])
                for note in staff if type(note) in [m2.note.Note, m2.chord.Chord]]

    @staticmethod
    def get_chord_or_note(chord_or_note):
        # currently takes root of chord
        # or bass of chord
        # does not consider major minor triad/seventh/etc.
        if type(chord_or_note) == m2.note.Note:
            return chord_or_note
        elif type(chord_or_note) == m2.chord.Chord:
            return m2.note.Note(chord_or_note.root(), duration=chord_or_note.duration)

    @staticmethod
    def list_to_stream(note_list):
        notes = [m2.note.Note(note[1], octave=int(note[2]), duration=m2.duration.Duration(round(note[0]*256)/256.0)) for note in note_list]
        stream = m2.stream.Stream()
        for note in notes:
            stream.append(note)
        return stream


class SonataNeuralNetwork(object):

    def __init__(self):
        self.before = 100
        self.after = 20
        self.ds = SupervisedDataSet(2 + 3 * self.before, 3 * self.after)
        self.net = buildNetwork(2 + 3 * self.before, 500, 1000, 250, 3 * self.after)

    def read(self, filename):
        piece = MIDIReader.parse(filename)
        for staff in piece['score']:
            notes = MIDIReader.get_notes(staff, piece['key_sigs'][0][0])
            self.append_dataset(notes, piece['time_sigs'][0])

    def append_dataset(self, notes, time_sig):
        for i in xrange(0, len(notes), self.before + self.after):
            if len(notes[i:]) < self.before + self.after:
                break
            inp = [time_sig[0], time_sig[1]]
            out = []
            for j in xrange(self.before):
                inp += [notes[i+j].duration.quarterLength, notes[i+j].pitchClass, notes[i+j].octave]
            for j in xrange(self.before, self.before + self.after):
                out += [notes[i+j].duration.quarterLength, notes[i+j].pitchClass, notes[i+j].octave]
            self.ds.addSample(inp, out)

    def train_network(self):
        trainer = BackpropTrainer(self.net, self.ds)
        while trainer.train() > 3:
            # print trainer.train()
            continue
        return self.net