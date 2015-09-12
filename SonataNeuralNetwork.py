from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import music21 as m2
from math import log, e
from numpy import std, mean


class MIDIReader(object):

    def __init__(self, filename):
        self.piece = {}
        self.piece['score'] = m2.converter.parse(filename)
        self.piece['key_sigs'] = [(sig.sharps, sig.getOffsetBySite(self.piece['score'][0]))
                                  for sig in list(self.piece['score'][0].getElementsByClass(m2.key.KeySignature))]
        self.piece['time_sigs'] = [(sig.beat, sig.denominator, sig.getOffsetBySite(self.piece['score'][0]))
                                   for sig in list(self.piece['score'][0].getElementsByClass(m2.meter.TimeSignature))]

    def get_notes(self):
        notes = []
        key_sig_adjust = {-7: 1, -6: 6, -5: -1, -4: 4, -3: -3, -2: 2, -1: -5, 0: 0, 1: 5, 2: -2, 3: 3, 4: -4, 5: 1, 6: -6, 7: -1}
        for staff in self.piece['score']:
            notes.append([MIDIReader.get_chord_or_note(note).transpose(key_sig_adjust[self.piece['key_sigs'][0][0]])
                          for note in staff if type(note) in [m2.note.Note, m2.chord.Chord]])
        return notes

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
        stream = m2.stream.Stream()
        for note in note_list:
            p = m2.pitch.Pitch()
            p.frequency = e ** note[1]
            stream.append(m2.note.Note(p, duration=m2.duration.Duration(round(note[0]*64)/64.0)))
        return stream


class SonataNeuralNetwork(object):

    def __init__(self, prev=5):
        self.ds = SupervisedDataSet(2 + (prev+1) * 2, 2)
        self.net = buildNetwork(2 + (prev+1) * 2, 50, 75, 25, 2)
        self.prev = prev
        self.corpus = []
        self.freq_err = []
        self.dur_err = []

    def read(self, filename):
        piece = MIDIReader(filename)
        self.corpus.append(piece)
        for notes in piece.get_notes():
            self.append_dataset(notes, piece.piece['time_sigs'][0])

    def append_dataset(self, notes, time_sig):
        for i in xrange(self.prev, len(notes) - 1):
            inp = [time_sig[0], time_sig[1]]
            for j in xrange(self.prev, -1, -1):
                inp += [notes[i-j].duration.quarterLength, log(notes[i-j].frequency)]
            self.ds.addSample(inp, (notes[i+1].duration.quarterLength, log(notes[i+1].frequency)))

    def train_network(self):
        trainer = BackpropTrainer(self.net, self.ds)
        while trainer.train() > 3800:
            # print trainer.train()
            continue
        return self.net

    def append_errors(self):
        for piece in self.corpus:
            for notes in piece.get_notes():
                for i in xrange(self.prev, len(notes) - 1):
                    inp = [piece.piece['time_sigs'][0][0], piece.piece['time_sigs'][0][1]]
                    for j in xrange(self.prev, -1, -1):
                        inp += [notes[i-j].duration.quarterLength, log(notes[i-j].frequency)]
                    dur, freq = self.net.activate(inp)
                    self.freq_err.append(freq - log(notes[i+1].frequency))
                    self.dur_err.append(dur - notes[i+1].duration.quarterLength)

    def get_error_vals(self):
        return (mean(self.dur_err), std(self.dur_err)), (mean(self.freq_err), std(self.freq_err))
