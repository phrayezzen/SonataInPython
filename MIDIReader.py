import music21 as m2


class MIDIReader(object):

    def __init__(self, filename):
        self.score = m2.converter.parse(filename)
        self.bass = self.score[-1]
        self.treble = self.score[-2]

    def get_notes(self):
        # currently takes root of chord
        # or bass of chord
        # does not consider major minor triad/seventh/etc.
        # TODO adjust notes based on key sig?
        tk, bk = self.get_key_sig()
        key_sig_adjust = {-7: 1, -6: 6, -5: -1, -4: 4, -3: -3, -2: 2, -1: -5, 0: 0, 1: 5, 2: -2, 3: 3, 4: -4, 5: 1, 6: -6, 7: -1}
        ta, ba = key_sig_adjust[tk], key_sig_adjust[bk]
        t = []
        b = []
        for note in self.treble:
            if type(note) == m2.note.Note:
                t.append(note.transpose(ta))
            elif type(note) == m2.chord.Chord:
                t.append(m2.note.Note(note.root(), duration=note.duration).transpose(ta))
        for note in self.bass:
            if type(note) == m2.note.Note:
                b.append(note.transpose(ba))
            elif type(note) == m2.chord.Chord:
                b.append(m2.note.Note(note.root(), duration=note.duration).transpose(ba))
        return t, b

    def get_key_sig(self):
        return (self.treble.getElementsByClass(m2.key.KeySignature)[0].sharps,
                self.bass.getElementsByClass(m2.key.KeySignature)[0].sharps)

    def get_time_sig(self):
        ts = self.treble.getElementsByClass(m2.meter.TimeSignature)[0]
        bs = self.bass.getElementsByClass(m2.meter.TimeSignature)[0]
        return (ts.beat, ts.denominator), (bs.beat, bs.denominator)
