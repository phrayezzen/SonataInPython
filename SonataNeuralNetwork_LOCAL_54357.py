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

    def process_notes(self):
        self.piece['treble'] = []
        self.piece['bass'] = []
        self.piece['chords'] = {}
        key_sig_adjust = {-7: 1, -6: 6, -5: -1, -4: 4, -3: -3, -2: 2, -1: -5, 0: 0, 1: 5, 2: -2, 3: 3, 4: -4, 5: 1, 6: -6, 7: -1}
        for note in self.piece['score'][0].flat:  # treble
            if type(note) in [m2.note.Note, m2.chord.Chord]:
                offset = note.getOffsetBySite(self.piece['score'][0].flat)
                self.piece['treble'].append((MIDIReader.get_chord_or_note(note)
                    .transpose(key_sig_adjust[self.piece['key_sigs'][0][0]]), offset))
                if offset % 1 == 0:
                    if offset in self.piece['chords']:
                        self.piece['chords'][offset] = m2.chord.Chord([note, self.piece['chords'][offset]])
                    else:
                        self.piece['chords'][offset] = m2.chord.Chord([note])

        for note in self.piece['score'][1].flat:  # bass
            if type(note) in [m2.note.Note, m2.chord.Chord]:
                offset = note.getOffsetBySite(self.piece['score'][1].flat)
                self.piece['bass'].append((MIDIReader.get_chord_or_note(note)
                    .transpose(key_sig_adjust[self.piece['key_sigs'][0][0]]), offset))
                if offset in self.piece['chords']:
                    self.piece['chords'][offset] = m2.chord.Chord([note, self.piece['chords'][offset]])
                else:
                    self.piece['chords'][offset] = m2.chord.Chord([note])

        return self.piece['treble'], self.piece['bass'], self.piece['chords']

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
            p.pitchClass = round(p.pitchClass)
            if p.pitchClass in [1, 6]:
                p.pitchClass += 1
            elif p.pitchClass in [3, 8, 10]:
                p.pitchClass -= 1
            stream.append(m2.note.Note(p, duration=m2.duration.Duration(round(note[0]*4)/4.0)))
        return stream


class SonataNeuralNetwork(object):

    def __init__(self, prev=5):
        # timsig beat, timsig denom, prev + curr dur/freq, prev 3 chords, bass note
        self.t_ds = SupervisedDataSet((prev+1) * 2 + 4, 2)
        self.t_net = buildNetwork((prev+1) * 2 + 4, 50, 75, 25, 2)
        self.t_freq_err = []
        self.t_dur_err = []

        self.b_ds = SupervisedDataSet((prev+1) * 2 + 4, 2)
        self.b_net = buildNetwork((prev+1) * 2 + 4, 50, 75, 25, 2)
        self.b_freq_err = []
        self.b_dur_err = []

        self.prev = prev
        self.corpus = []

    def read(self, filename):
        piece = MIDIReader(filename)
        self.corpus.append(piece)
        treb, bass, chords = piece.process_notes()
        self.append_dataset(treb, piece)
        self.append_dataset(bass, piece, False)

    def append_dataset(self, staff, piece, treble=True):
        for i in xrange(self.prev, len(staff) - 1):
            inp = self.get_input(staff, i, piece, treble)
            ds = self.t_ds if treble else self.b_ds
            ds.addSample(inp, (staff[i+1][0].duration.quarterLength, log(staff[i+1][0].frequency)))

    def train_network(self):
        t_trainer = BackpropTrainer(self.t_net, self.t_ds)
        while t_trainer.train() > 1.5:
            print t_trainer.train()
            continue
        b_trainer = BackpropTrainer(self.b_net, self.b_ds)
        while b_trainer.train() > 1.5:
            print b_trainer.train()
            continue
        return self.t_net, self.b_net

    def append_errors(self):
        for piece in self.corpus:
            staff = piece.piece['treble']
            for i in xrange(self.prev, len(staff) - 1):
                inp = self.get_input(staff, i, piece)
                dur, freq = self.t_net.activate(inp)
                self.t_freq_err.append(freq - log(staff[i+1][0].frequency))
                self.t_dur_err.append(dur - staff[i+1][0].duration.quarterLength)

            staff = piece.piece['bass']
            for i in xrange(self.prev, len(staff) - 1):
                inp = self.get_input(staff, i, piece)
                dur, freq = self.b_net.activate(inp)
                self.b_freq_err.append(freq - log(staff[i+1][0].frequency))
                self.b_dur_err.append(dur - staff[i+1][0].duration.quarterLength)

    def get_input(self, staff, i, piece, treble=True):
        inp = []
        for j in xrange(self.prev, -1, -1):
            inp += [staff[i-j][0].duration.quarterLength, log(staff[i-j][0].frequency)]

        offset = int(staff[i][1])
        log_freq = log(staff[i][0].frequency)
        other_freq = piece.piece['score'][1 if treble else 0].getElementsByOffset(offset)
        if len(other_freq) < 1:
            other_freq = log_freq
        elif type(other_freq[0]) == m2.note.Note:
            other_freq = other_freq[0].frequency
        elif type(other_freq[0]) == m2.chord.Chord:
            other_freq = other_freq[0].root().frequency
        else:
            other_freq = log_freq

        inp += [log(piece.piece['chords'][offset-2].root().frequency) if offset-2 in piece.piece['chords'] else log_freq,
                log(piece.piece['chords'][offset-1].root().frequency) if offset-1 in piece.piece['chords'] else log_freq,
                log(piece.piece['chords'][offset].root().frequency) if offset in piece.piece['chords'] else log_freq,
                other_freq]
        return inp

    def get_error_vals(self):
        return ((mean(self.t_dur_err), std(self.t_dur_err)), (mean(self.t_freq_err), std(self.t_freq_err)),
                (mean(self.b_dur_err), std(self.b_dur_err)), (mean(self.b_freq_err), std(self.b_freq_err)))
