"""bin/draw --text: handwritten English, read by the letter network (examples/letters.anvil, EMNIST)
with the help of a word list."""
import json
import os
import sys
import threading
import urllib.request

import numpy as np
import pytest

from util import ROOT

sys.path.insert(0, os.path.join(ROOT, "examples", "draw"))
import sample_strokes  # noqa: E402
import server  # noqa: E402
import text_reader  # noqa: E402
from sample_strokes import write  # noqa: E402
from text_reader import Dictionary, segment  # noqa: E402

WEIGHTS = os.path.join(ROOT, "examples", "letters.weights")
need_weights = pytest.mark.skipif(not os.path.exists(WEIGHTS),
                                  reason="needs examples/letters.weights (bin/anvil run examples/letters.anvil, with EMNIST)")


def shape(lines):
    """Characters per word, per line (pieces that surely belong together)."""
    return [[len(w.pieces) for w in words] for words in lines]


def test_lines_words_and_letters():
    assert shape(segment(write("hello world\nnew line", seed=1))) == [[5, 5], [3, 4]]
    assert segment([]) == [] and segment([[]]) == []


def test_strokes_that_make_one_letter():
    # the dot of an i, the bar of a t, the three strokes of an E, the stem and arch of an h
    lines = segment(write("it Eh", seed=2))
    assert shape(lines) == [[2, 2]]
    assert [len(p.strokes) for p in lines[0][0].pieces] == [2, 2]
    # the arches of an m only touch: one letter or two is left to the network and the word list
    m = segment([s for s in sample_strokes.GLYPHS["m"]])[0][0]
    assert len(m.pieces) == 2 and m.links


def test_capitals_on_several_lines():
    """The bars of T, E, I are at the very top and bottom of their line; they must not join the lines."""
    assert shape(segment(write("TIE IT\nTIE", seed=3))) == [[3, 2], [3]]


def test_punctuation():
    words = [w for line in segment(write("Hello, world. It's", seed=0)) for w in line]
    assert [w.marks for w in words] == [",", ".", ""]
    assert words[2].apostrophe == 2 and len(words[2].pieces) == 3


def test_case_from_size():
    lines = segment(write("Coco", seed=0))
    big, small = lines[0][0].pieces[0], lines[0][0].pieces[2]
    assert not big.small and small.small
    # all one size: size can't tell, so it says nothing
    assert all(not p.small and p.full == 0 for p in segment(write("coco", seed=0))[0][0].pieces)


def test_dictionary():
    d = Dictionary()
    def onehot(word, sure=0.9):
        q = np.full((len(word), 26), (1 - sure) / 25)
        q[np.arange(len(word)), [ord(c) - 97 for c in word]] = sure
        return np.log(q)
    assert d.best(onehot("hello"))[0] == "hello"
    assert d.best(onehot("words"))[0] == "words"                      # word + s: the list has only "word"
    assert d.best(onehot("played"))[0] == "played"
    q = onehot("hell0"[:4] + "o")
    q[4] = np.log(np.full(26, 1 / 26))                                 # the last letter unreadable
    assert d.best(q)[0].startswith("hell")
    assert "en" not in d.by_len[2] and "the" in d.by_len[3]           # short words: only common ones


@pytest.fixture(scope="module")
def reader():
    return text_reader.TextReader(WEIGHTS)


@need_weights
def test_reads_printed_words(reader):
    phrases = ["hello world", "The quick brown fox", "machine learning", "call 911 now",
               "I love Pittsburgh", "she played with cats", "well done, friend."]
    right = total = 0
    for text in phrases:
        for seed in range(3):
            got = reader.read(write(text, seed=seed))["text"].split()
            right += sum(a == b for a, b in zip(text.split(), got))
            total += len(text.split())
    assert right >= 0.95 * total, f"{right} of {total}"


@need_weights
def test_context_fixes_what_the_network_confuses(reader):
    out = reader.read(write("hello world", seed=0))
    raw = " ".join(w.raw for w in out["lines"][0])
    assert out["text"] == "hello world"
    assert raw != "hello world"                    # alone, the network reads l as 1 and o as 0
    assert reader.read(write("call 911", seed=1))["text"] == "call 911"     # and a number stays a number


@need_weights
def test_capitals(reader):
    assert reader.read(write("STOP NOW", seed=0))["text"] == "STOP NOW"
    assert reader.read(write("So good", seed=0))["text"] == "So good"


@need_weights
def test_http_text(reader):
    server.Handler.text, server.Handler.text_missing = reader, None
    httpd = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/text").read().decode()
        assert "<canvas" in page and "/read_text" in page

        def read(strokes):
            req = urllib.request.Request(base + "/read_text", data=json.dumps({"strokes": strokes}).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req))
        out = read(write("hello", seed=0))
        assert out["text"] == "hello"
        (word,), = out["lines"]
        assert word["raw"] != "hello" and len(word["chars"]) == 5
        c = word["chars"][0]
        assert c["text"] == "h" and len(c["image"]) == 784 and len(c["box"]) == 4 and len(c["top"]) == 3
        assert read([]) == {"empty": True}
        server.Handler.text, server.Handler.text_missing = None, "train it"
        assert read(write("hi", seed=0)) == {"missing": "train it"}
    finally:
        server.Handler.text = server.Handler.text_missing = None
        httpd.shutdown()
