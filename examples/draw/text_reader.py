"""Reading handwritten English: strokes → lines → words → characters → the letter network
(examples/letters_model.anvil, trained by examples/letters.anvil on EMNIST) → words.

1. Lines: strokes are grouped by height on the page.
2. Characters: in a line, strokes that overlap horizontally belong to one character (the dot of
   an i, the bar of a t, the three strokes of an E). Strokes that only touch may be one letter (the
   arches of an m) or two letters written close together; both readings are tried (step 5).
3. Words: a gap much wider than the gaps between letters starts a new word.
4. Each character is drawn again the way EMNIST's are (scaled to fill the frame, centered, with
   EMNIST's stroke width) and read by the network: a probability for each of 0-9, A-Z, a-z.
5. Context. Each word is matched against a word list using the network's probabilities, so a
   misread letter is corrected when the word is an English word (l/1, o/0, …); a name that is not
   in the list is kept as read, and a number as a number. Of the ways to join touching strokes, the
   one whose reading is most probable wins.
6. Case. A letter like a/A or n/N says its case by its shape; c/C, o/O, s/S, … only by their size
   (how far they rise above the line). The letters vote: CAPITALS, Capitalized, or lowercase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from server import EXAMPLES, render

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTERS_MODEL = os.path.join(EXAMPLES, "letters_model.anvil")
LETTERS_WEIGHTS = os.path.join(EXAMPLES, "letters.weights")
BOX, RADIUS = 20.0, 1.45                 # EMNIST's characters: ~23 px across, with ~135 units of ink
SAME_SHAPE = set("ckopsuvwxz")        # letters whose two cases differ only in size
ASCENDERS = set("bdfhijklt")          # lowercase letters that rise as high as capitals
WORDS_FILE = "/usr/share/dict/words"
COMMON = """the of and to a in is it you that he was for on are with as I his they be at one have this
from or had by not word but what some we can out other were all there when up use your how said an
each she which do their time if will way about many then them write would like so these her long
make thing see him two has look more day could go come did number sound no most people my over
know water than call first who may down side been now find any new work part take get place made
live where after back little only round man year came show every good me give our under name very
through just form sentence great think say help low line differ turn cause much mean before move
right boy old too same tell does set three want air well also play small end put home read hand
port large spell add even land here must big high such follow act why ask men change went light
kind off need house picture try us again animal point mother world near build self earth father
head stand own page should country found answer school grow study still learn plant cover food sun
four between state keep eye never last let thought city tree cross farm hard start might story saw
far sea draw left late run don't while press close night real life few north open seem together
next white children begin got walk example ease paper group always music those both mark often
letter until mile river car feet care second book carry took science eat room friend began idea
fish mountain stop once base hear horse cut sure watch color face wood main enough plain girl usual
young ready above ever red list though feel talk bird soon body dog family direct pose leave song
measure door product black short numeral class wind question happen complete ship area half rock
order fire south problem piece told knew pass since top whole king space heard best hour better
true during hundred five remember step early hold west ground interest reach fast verb sing listen
six table travel less morning ten simple several vowel toward war lay against pattern slow center
love person money serve appear road map rain rule govern pull cold notice voice unit power town
fine certain fly fall lead cry dark machine note wait plan figure star box noun field rest correct
able pound done beauty drive stood contain front teach week final gave green oh quick develop ocean
warm free minute strong special mind behind clear tail produce fact street inch multiply nothing
course stay wheel full force blue object decide surface deep moon island foot system busy test
record boat common gold possible plane age dry wonder laugh thousand ago ran check game shape yes
hot miss brought heat snow tire bring distant fill east paint language among hello world quick
brown fox jumps lazy machine learning network neural deep data model train language""".split()
# the word list's short words are mostly obscure ("en", "aal", "ers"), and a short word has little
# evidence to go on: these are the only words of 1-3 letters it knows
SHORT = set("""a i am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we ad ah
ax ex ma pa ye yo add age ago aid aim air all and any ape apt arc are arm art ash ask ate awe axe bad
bag ban bar bat bay bed bee beg bet bid big bin bit bow box boy bud bug bum bun bus but buy bye cab
can cap car cat cop cow cry cub cue cup cut dad dam day den dew did die dig dim dip dog dot dry due
dug dye ear eat egg ego elf elk elm emu end era eve ewe eye fad fan far fat fax fed fee few fig fin
fit fix flu fly foe fog for fox fry fun fur gag gap gas gel gem get gig god got gum gun gut guy gym
had ham has hat hay hen her hey hid him hip his hit hog hop hot how hub hue hug hum hut ice icy ill
imp ink inn ion its ivy jab jam jar jaw jay jet jig job jog joy jug keg key kid kin kit lab lad lag
lap law lay led leg let lid lie lip lit log lot low mad man map mat may men met mix mob mom mop mud
mug nab nag nap net new nil nod nor not now nun nut oak oar oat odd off oil old one opt orb ore our
out owe owl own pad pal pan par pat paw pay pea peg pen pep per pet pew pie pig pin pit ply pod pop
pot pro pry pub pun pup put rag ram ran rap rat raw ray red rib rid rig rim rip rob rod rot row rub
rug rum run rut rye sad sag sap sat saw say sea see set sew she shy sin sip sir sis sit six ski sky
sly sob sod son sow soy spa spy sub sue sum sun tab tag tan tap tar tax tea ten the tie tin tip toe
ton too top tow toy try tub tug two urn use van vat vet via vow wag war was wax way web wed wet who
why wig win wit woe wok won wow yak yam yap yaw yes yet you zap zip zoo""".split())


@dataclass
class Char:
    strokes: list
    box: tuple                            # x0, y0, x1, y1
    base: float = 0.0                     # the baseline of its line
    full: float = 0.0                     # how far capitals rise above the baseline on this page (0: can't tell)
    mark: str = ""                        # punctuation: ".", "," or "'"
    probs: np.ndarray = None
    image: np.ndarray = None
    text: str = ""

    @property
    def small(self):
        """Lowercase-sized: it rises less above the baseline than capitals do."""
        return bool(self.full) and self.base - self.box[1] < 0.75 * self.full


def join(pieces):
    strokes = [s for p in pieces for s in p.strokes]
    c = Char(strokes, bbox(strokes), pieces[0].base, pieces[0].full)
    c.first = pieces[0]
    return c


@dataclass
class Word:
    pieces: list                          # strokes that surely belong to one character
    links: set                            # pairs (i, j) of pieces that touch: one letter, or two?
    marks: str = ""                       # punctuation after the word
    apostrophe: int = -1                  # an apostrophe before pieces[apostrophe] (it's, don't)
    chars: list = field(default_factory=list)   # the characters, as read
    caps: bool = None                     # written in capitals (None: can't tell, as in "so")
    text: str = ""
    raw: str = ""                         # each character's best guess, before context


def bbox(strokes):
    pts = np.concatenate([np.asarray(s, dtype=float).reshape(-1, 2) for s in strokes])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def dense(stroke, step):
    """The points of a stroke, with more added along long segments: every `step` apart."""
    p = np.asarray(stroke, dtype=float).reshape(-1, 2)
    out = [p[:1]]
    for a, b in zip(p, p[1:]):
        k = max(int(np.hypot(*(b - a)) / step), 1)
        out.append(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.concatenate(out)


def segment(strokes):
    """Lines (top to bottom) of words (left to right), from the strokes."""
    strokes = [s for s in strokes if len(s)]
    if not strokes:
        return []
    boxes = [bbox([s]) for s in strokes]
    heights = np.array([b[3] - b[1] for b in boxes])
    h = max(float(np.percentile(heights, 90)), 1.0)         # about a capital letter's height
    # lines, from the tall strokes (the bars of a T or an E are at the very top or bottom of their
    # line, and would bridge the gap to the next): sorted by height on the page, split at big jumps
    tall = [i for i in range(len(strokes)) if heights[i] >= 0.35 * h] or list(range(len(strokes)))
    center = lambda i: (boxes[i][1] + boxes[i][3]) / 2
    lines = []
    for i in sorted(tall, key=center):
        if not lines or center(i) - center(lines[-1][-1]) > 0.6 * h:
            lines.append([])
        lines[-1].append(i)
    spans = [(min(boxes[i][1] for i in ln), max(boxes[i][3] for i in ln)) for ln in lines]
    for i in set(range(len(strokes))) - set(tall):                 # bars, dots, marks: the nearest line
        c = center(i)
        k = min(range(len(lines)), key=lambda k: max(spans[k][0] - c, c - spans[k][1], 0.0))
        lines[k].append(i)
    lines = [pieces([strokes[i] for i in ln], [boxes[i] for i in ln], h) for ln in lines]
    size([p for p, _ in lines], h)
    return [words(p, links) for p, links in lines]


def pieces(strokes, boxes, h):
    """A line's strokes grouped into pieces of characters, left to right, and the pairs of pieces
    that may be one character. Strokes that overlap horizontally almost completely (each stroke's
    box widened a little) are one: the bar of a t, the stem and arch of an h, the three strokes of
    an E; and a dot belongs to the letter under it. Strokes that overlap a little or touch may be
    one letter (the arches of an m) or two letters written close together: those are links."""
    tall = [b[3] - b[1] for b in boxes if b[3] - b[1] >= 0.35 * h]
    pad, near = 0.025 * h, 0.15 * float(np.median(tall) if tall else h)    # near: about a sixth of a small letter
    groups = [([strokes[i]], [boxes[i][0] - pad, boxes[i][1], boxes[i][2] + pad, boxes[i][3]])
              for i in sorted(range(len(strokes)), key=lambda i: boxes[i][0])]
    dotted = set()                                                     # pieces with a dot: an i or a j, whole
    overlap = lambda ba, bb: min(ba[2], bb[2]) - max(ba[0], bb[0])
    dot = lambda g: max(g[1][2] - g[1][0] - 2 * pad, g[1][3] - g[1][1]) < 0.22 * h

    def merge(same):
        changed = True
        while changed:
            changed = False
            for a in range(len(groups)):
                for b in range(len(groups)):
                    if a != b and same(groups[a], groups[b]):
                        (sa, ba), (sb, bb) = groups[a], groups[b]
                        groups[min(a, b)] = (sa + sb, [min(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]), max(ba[3], bb[3])])
                        del groups[max(a, b)]
                        changed = True
                        break
                if changed:
                    break
    merge(lambda ga, gb: overlap(ga[1], gb[1]) > 0.6 * min(ga[1][2] - ga[1][0], gb[1][2] - gb[1][0]))
    for d in [g for g in groups if dot(g)]:                            # a dot belongs to the letter under it
        x = (d[1][0] + d[1][2]) / 2
        under = [g for g in groups if not dot(g) and overlap(d[1], g[1]) > 0 and d[1][3] < g[1][1] + 0.1 * h]
        if under:
            g = min(under, key=lambda g: abs((g[1][0] + g[1][2]) / 2 - x))
            g[0].extend(d[0])
            dotted.add(id(g[0]))
            g[1][:] = [min(g[1][0], d[1][0]), min(g[1][1], d[1][1]), max(g[1][2], d[1][2]), g[1][3]]
            groups.remove(d)
    groups.sort(key=lambda g: g[1][0])
    points = [np.concatenate([dense(s, near / 2) for s in g[0]]) for g in groups]
    links = set()
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            ba, bb = groups[a][1], groups[b][1]
            if max(ba[2], bb[2]) - min(ba[0], bb[0]) > 1.2 * h + 2 * pad or {id(groups[a][0]), id(groups[b][0])} & dotted:
                continue
            if overlap(ba, bb) > 0 or (((points[a][:, None, :] - points[b][None, :, :]) ** 2).sum(-1).min() < near * near):
                links.add((a, b))
    return [Char(g[0], bbox(g[0])) for g in groups], links


def size(lines, h):
    """Each piece's baseline and the page's capital height, and which pieces are punctuation. A
    letter's size is how far it rises above its line's baseline (so the tails of g, p, y don't
    count), compared with the tallest letters on the page: capitals and ascenders. When all the
    letters are about the same size (ALL CAPS, or "so so"), size says nothing."""
    rise = []
    for chars in lines:
        if chars:
            base = float(np.median([c.box[3] for c in chars]))
            for c in chars:
                c.base = base
                rise.append(base - c.box[1])
    top = max(rise) if rise else h
    full = top if rise and top > 1.3 * np.percentile(rise, 25) else 0.0
    for chars in lines:
        for c in chars:
            c.full = full
            w, ht = c.box[2] - c.box[0], c.box[3] - c.box[1]
            if max(w, ht) < 0.22 * (full or h):                  # a dot, a comma, an apostrophe
                below = c.box[3] - c.base                           # a comma's tail hangs below the line
                c.mark = "'" if c.base - c.box[1] > 0.6 * (full or h) else "," if below > 0.06 * (full or h) else "."


def word_gap(gaps, height):
    """How wide a gap starts a new word: at the biggest jump between the sorted gaps (the gaps
    between letters below it, the spaces between words above), if there is a clear one."""
    gaps = sorted([0.15 * height] + gaps)                       # as if there were a usual letter gap
    best, split = 1.6, np.inf
    for a, b in zip(gaps, gaps[1:]):
        if b > 0.4 * height and b / max(a, 0.05 * height) > best:
            best, split = b / max(a, 0.05 * height), max(np.sqrt(max(a, 0.05 * height) * b), 0.4 * height)
    return split


def words(chars, links):
    """Pieces into words: a gap much wider than the gaps between letters starts a new word."""
    letters = [i for i, c in enumerate(chars) if not c.mark]
    height = float(np.median([chars[i].box[3] - chars[i].box[1] for i in letters])) if letters else 1.0
    split = word_gap([chars[b].box[0] - chars[a].box[2] for a, b in zip(letters, letters[1:])], height)
    out, cur, prev, edge, index = [], None, None, None, {}
    for i, c in enumerate(chars):
        if c.mark:
            after = next((chars[k] for k in range(i + 1, len(chars)) if not chars[k].mark), None)
            if c.mark == "'" and cur is not None and not cur.marks and after is not None and after.box[0] - c.box[2] <= split:
                cur.apostrophe = len(cur.pieces)
                edge = max(edge, c.box[2])
            elif cur is not None:
                cur.marks += c.mark
            continue
        if cur is None or cur.marks or (c.box[0] - edge > split and (prev, i) not in links):
            cur = Word([], set())
            out.append(cur)
        index[i] = (cur, len(cur.pieces))
        cur.pieces.append(c)
        prev, edge = i, c.box[2]
    for a, b in links:
        if a in index and b in index and index[a][0] is index[b][0]:
            index[a][0].links.add((index[a][1], index[b][1]))
    return out


class Dictionary:
    """Words by length, as arrays of letter indices, for scoring every word of a length at once."""

    def __init__(self, path=WORDS_FILE):
        words = set(COMMON)
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                words |= {w.strip().lower() for w in f}
        words = {w for w in words if w.isascii() and w.isalpha() and (len(w) > 3 or w in SHORT)}
        self.common = {w.lower() for w in COMMON}
        self.by_len = {}
        for w in words:
            self.by_len.setdefault(len(w), []).append(w)
        self.idx = {n: np.array([[ord(ch) - 97 for ch in w] for w in ws], dtype=np.int32)
                    for n, ws in self.by_len.items()}
        self.bonus = {n: np.array([2.0 if w in self.common else 0.0 for w in ws]) for n, ws in self.by_len.items()}
        self.cache = {}

    SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "ly")    # the word list has "word" but not "words"

    def best(self, logq: np.ndarray):
        """The word (and its score) that the per-letter log-probabilities logq[position, letter]
        like best: a word in the list, or one with a suffix (word+s, play+ed, ...), which costs a
        little; common words get a small bonus."""
        n = len(logq)
        word, score = self.lookup(logq)
        for suffix in self.SUFFIXES:
            k = len(suffix)
            if n - k < 2:
                continue
            stem, s = self.lookup(logq[:n - k])
            if stem is not None:
                s += sum(float(logq[n - k + i, ord(ch) - 97]) for i, ch in enumerate(suffix)) - 1.0
                if s > score:
                    word, score = stem + suffix, s
        return word, score

    def lookup(self, logq: np.ndarray):
        n = len(logq)
        if n not in self.idx:
            return None, -np.inf
        key = logq.tobytes()
        if key not in self.cache:
            scores = logq[np.arange(n)[None, :], self.idx[n]].sum(1) + self.bonus[n]
            k = int(np.argmax(scores))
            self.cache = {key: (self.by_len[n][k], float(scores[k]))} if len(self.cache) > 4096 else self.cache
            self.cache[key] = self.by_len[n][k], float(scores[k])
        return self.cache[key]


class TextReader:
    BEAM = 12                             # ways of dividing a word into characters, kept while searching
    RUN = 4                               # the most pieces one character may have

    def __init__(self, weights=LETTERS_WEIGHTS, dictionary: Dictionary | None = None):
        import anvil
        source = (f'use "{LETTERS_MODEL}"\n'
                  f"net = LetterNet()\n"
                  f'found = load(net, "{weights}")\n'
                  f"fn read(image: [28, 28]):\n"
                  f"    p = softmax(net(image.reshape(1, 1, 28, 28)))\n"
                  f"    return p[0], found\n")
        self.fn = anvil.function(source, name="read")
        _, found = self.fn(np.zeros((28, 28), np.float32))
        if found != 1:
            raise RuntimeError(f"cannot load {weights}: run `bin/anvil run examples/letters.anvil` to make it")
        self.dictionary = dictionary or Dictionary()

    def classify(self, c: Char):
        c.image = render(c.strokes, box=BOX, radius=RADIUS, by_mass=False)
        c.probs = self.fn(c.image.astype(np.float32))[0]

    def read(self, strokes) -> dict:
        lines = segment(strokes)
        start = True                                                   # of a sentence
        for words in lines:
            for w in words:
                self.read_word(w, start)
                start = w.marks.endswith(".")
        # a word whose letters don't say their case (so, OX) is written like the words near it
        every = [w for words in lines for w in words]
        for k, w in enumerate(every):
            if w.caps is None:
                near = sorted((abs(j - k), v.caps) for j, v in enumerate(every) if v.caps is not None)
                if near and near[0][1]:
                    w.text = w.text.upper()
            if w.apostrophe >= 0:
                at = sum(1 for c in w.chars if w.pieces.index(c.first) < w.apostrophe)
                w.text = w.text[:at] + "'" + w.text[at:]
        text = "\n".join(" ".join(w.text + w.marks for w in words) for words in lines)
        return {"text": text, "lines": lines}

    def read_word(self, w: Word, start=True):
        """Divide the word's pieces into characters. Each character is a run of pieces next to each
        other that are linked; a beam search keeps the divisions whose characters the network reads
        most surely, and of those, the one whose reading of the whole word is most probable wins. A
        character much wider than a capital is tall is probably two."""
        n = len(w.pieces)
        chars = {}

        def char(i, j):                                                # pieces i..j-1 as one character
            if (i, j) not in chars:
                ok = j - i == 1 or (j - i <= self.RUN and connected(i, j))
                c = join(w.pieces[i:j]) if ok else None
                if c is not None:
                    self.classify(c)
                    wide = 8 * max(0.0, (c.box[2] - c.box[0]) / (c.full or c.box[3] - c.box[1] or 1.0) - 0.75)
                    c.fit = float(np.log(max(c.probs[:10].max(), (c.probs[10:36] + c.probs[36:]).max()))) - wide
                    c.wide = wide
                chars[i, j] = c
            return chars[i, j]

        def connected(i, j):
            seen, todo = {i}, [i]
            while todo:
                a = todo.pop()
                for b in range(i, j):
                    if b not in seen and ((a, b) in w.links or (b, a) in w.links):
                        seen.add(b)
                        todo.append(b)
            return len(seen) == j - i

        beams = [[(0.0, [])]] + [[] for _ in range(n)]                 # beams[j]: divisions of pieces 0..j-1
        for j in range(1, n + 1):
            options = []
            for i in range(max(0, j - self.RUN), j):
                c = char(i, j)
                if c is not None:
                    options += [(score + c.fit, cs + [c]) for score, cs in beams[i]]
            beams[j] = sorted(options, key=lambda o: -o[0])[:self.BEAM]
        best = None
        for _, cs in beams[n]:
            score, text, caps = self.decide(cs, start)
            score -= sum(c.wide for c in cs)
            if best is None or score > best[0]:
                best = score, text, cs, caps
        _, w.text, w.chars, w.caps = best
        w.raw = "".join(CLASSES[int(np.argmax(c.probs))] for c in w.chars)
        for c, ch in zip(w.chars, w.text):
            c.text = ch

    def decide(self, chars, start=True):
        """The most probable reading of the characters, and its log-probability. Three readings
        compete: the dictionary word the network's probabilities like best, the letters as read one by
        one (for names and words not in the list), and a number. Then the case: CAPITALS,
        Capitalized or lowercase, by a vote of the letters; a capital in the middle of a sentence
        needs to be clearer."""
        P = np.array([c.probs for c in chars])                         # [n, 62]
        n = len(P)
        digits, upper, lower = P[:, :10], P[:, 10:36], P[:, 36:62]
        letters = upper + lower                                        # a letter, either case
        either = np.concatenate([digits, letters], 1)                  # [n, 36]: 0-9 a-z
        log = lambda a: np.log(np.maximum(a, 1e-6))
        readings = []
        word, score = self.dictionary.best(log(letters))
        if word is not None:
            readings.append((score, word))
        known = word
        raw = "".join("0123456789abcdefghijklmnopqrstuvwxyz"[i] for i in either.argmax(1))
        mixed = 2.0 if any(ch.isdigit() for ch in raw) and any(ch.isalpha() for ch in raw) else 0.0
        readings.append((float(log(either.max(1)).sum()) - 0.8 * n - 1.0 - mixed, raw))  # not a word we know
        readings.append((float(log(digits.max(1)).sum()) - 1.0, "".join(str(int(i)) for i in digits.argmax(1))))
        score, text = max(readings)
        # the evidence that each letter is a capital: its shape, for letters whose capital has its own
        # (not c/C, o/O, s/S, ...), though only so much, since EMNIST's letters are all scaled to the
        # same size; and its size: a small letter is never a capital, and only capitals and ascenders
        # (b, d, f, ...) rise to the full height
        votes = []                                                     # None: this letter can't tell
        for c, ch in zip(chars, text):
            if not ch.isalpha() or (ch in SAME_SHAPE and not c.full):
                votes.append(None)
                continue
            i = ord(ch) - 97
            e = 0.0 if ch in SAME_SHAPE else float(np.clip(np.log(max(c.probs[10 + i], 1e-6) / max(c.probs[36 + i], 1e-6)), -3, 3))
            if c.full:
                e += -3.5 if c.small else 0.0 if ch in ASCENDERS else 1.5
            votes.append(e)
        later = [v > 0 for v in votes[1:] if v is not None]
        if len(text) == 1:
            later = [v > 0 for v in votes if v is not None]
        caps = all(later) and (votes[0] is None or votes[0] > 0) if later else None   # None: can't tell
        if caps:
            text = text.upper()
        elif votes[0] is not None and votes[0] - (2.0 if text == known and not start else 0.0) > 0:
            text = text[0].upper() + text[1:]
        return score, text, caps
