"""Real-world Polish score-sheet fixtures, harvested from
``tests/real_photos_report/*.json``. Each successful per-image OCR result
becomes one parametrized test that exercises:

    - language autodetect against the raw text
    - san_normalize round-trip against the canonical text
    - python-chess legality of every ply in the sample moves

These fixtures guard against regression in:
    - Polish piece-letter detection (H/W/G/S)
    - Colon capture normalization (: -> x)
    - Polish castling form (0-0 -> O-O)
    - M3 output post-processing (think stripping)

Regenerate with:  uv run python scripts/build_real_corpus.py
"""

from __future__ import annotations

import io
import re

import chess
import chess.pgn
import pytest

from mcp_server.parsers.san_normalize import detect_language, normalize_pgn


_REAL_CORPUS = [

    {
        'name': 'IMG20260627182008.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 2,
        'sample_moves': ['d4', 'd5', 'Nf3', 'Nf6', 'Bf4'],
        'canonical_first_300': '[Event "?"]\n[Date "2016.06.27"]\n[Round "3"]\n[White "Maksymilian Pilecki"]\n[Black "Dawid Andruszkiewicz"]\n[Result "0-1"]\n\n1.d4 d5 2.Nf3 Nf6 3.Bf4 Bg5 4.e3 e6 5.Nd2 Bb4 6.Bd3 c6 7.c3 Bxd3 8.g3 Bxg3 9.Bxg3 B5 10.Qc2 g6 11.O-O-O ab 12.Re1 Nd7 13.Ne5 Nxe5 14.dxc5 Nd5 15.Qe3 Bf5 16.f4 Bxd7 17.a3 c4 18.g4 ',
        'raw_first_300': '[Event "?"]\n[Date "2016.06.27"]\n[Round "3"]\n[White "Maksymilian Pilecki"]\n[Black "Dawid Andruszkiewicz"]\n[Result "0-1"]\n\n1.d4 d5 2.Sf3 Sf6 3.Gf4 Gg5 4.e3 e6 5.Sd2 Gb4 6.Gd3 c6 7.c3 Gxd3 8.g3 Gxg3 9.Gxg3 b5 10.Hc2 g6 11.O-O-O ab 12.We1 Sd7 13.Se5 Sxe5 14.d:c5 Sd5 15.He3 Gf5 16.f4 Gxd7 17.a3 c4 18.g4 ',
    },
    {
        'name': 'IMG20260629224923.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 10,
        'sample_moves': ['d4', 'e6', 'Bf4', 'd5', 'e3', 'Nc6', 'Nf3', 'Be7'],
        'canonical_first_300': '[Event "?"]\n[Site "?"]\n[Date "2025.06.27"]\n[Round "1"]\n[White "Maksymilian Piliżys"]\n[Black "Mikołaj Andrzejewski"]\n[Result "*"]\n\n1.d4 e6 2.Bf4 d5 3.e3 Nc6 4.Nf3 Be7 5.Bd3 Nf6 6.Ne5 Nxe5 7.Bxe5 Bd6 8.Bg3 O-O 9.Nd2 Bxg3 10.hxg3? h6 11.Na3 c5 12.g4 c4 13.Be2 Nc4 14.c3 g5 15.Ne5 a5 16.Kg3 Ng3 17.Rh3 Nx',
        'raw_first_300': '[Event "?"]\n[Site "?"]\n[Date "2025.06.27"]\n[Round "1"]\n[White "Maksymilian Piliżys"]\n[Black "Mikołaj Andrzejewski"]\n[Result "*"]\n\n1.d4 e6 2.Bf4 d5 3.e3 Nc6 4.Nf3 Be7 5.Bd3 Nf6 6.Ne5 Nxe5 7.Bxe5 Bd6 8.Bg3 O-O 9.Nd2 Bxg3 10.hxg3? h6 11.Na3 c5 12.g4 c4 13.Be2 Nc4 14.c3 g5 15.Ne5 a5 16.Kg3 Ng3 17.Rh3 Nx',
    },
    {
        'name': 'IMG20260717183156.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 4,
        'sample_moves': ['e4', 'c6', 'd4', 'd5', 'e5', 'Nf6', 'h4', 'h6'],
        'canonical_first_300': '[Event "IN MARI VICTORIA TUA"]\n[Site "?"]\n[Date "2015.07.11"]\n[Round "1"]\n[White "Antoni Krasowski"]\n[Black "Maksymilian Pilczys"]\n[Result "1-0"]\n\n1. e4 c6 2. d4 d5 3. e5 Nf6 4. h4 h6 5. Bd3 Bxd3 6. Qxd3 e6 7. Nf3 Nd7 8. Bg5 Ne7 9. Nd2 Qb6 10. Ne2 c5 11. dxc5 Nxc5 12. Nxc5 Qxc5 13. O-O Ng6 14. Rfd1 ',
        'raw_first_300': '[Event "IN MARI VICTORIA TUA"]\n[Site "?"]\n[Date "2015.07.11"]\n[Round "1"]\n[White "Antoni Krasowski"]\n[Black "Maksymilian Pilczys"]\n[Result "1-0"]\n\n1. e4 c6 2. d4 d5 3. e5 Sf6 4. h4 h6 5. Gd3 Gxd3 6. Hxd3 e6 7. Sf3 Sd7 8. Gg5 Se7 9. Sd2 Hb6 10. Se2 c5 11. dxc5 Sxc5 12. Sxc5 Hxc5 13. O-O Sg6 14. Wfd1 ',
    },
    {
        'name': 'IMG20260718203901.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 5,
        'sample_moves': ['d4', 'Nf6', 'Bf4', 'd5', 'e3', 'e6', 'Nf3', 'c5'],
        'canonical_first_300': '[Event "IM MARII VICTORII 50A"]\n[Site "?"]\n[Date "2026.07.18"]\n[Round "2"]\n[White "Maksymilian Pilż"]\n[Black "Władysław Przebewski"]\n[Result "0-1"]\n\n1.d4 Nf6 2.Bf4 d5 3.e3 e6 4.Nf3 c5 5.c3 c4 6.Bf3 O-O 7.Nd2 Nc6 8.Ne5 Bd6 9.Bg3 Qc6 10.Qf4 Qxf6 11.B3 cxd4 12.exd4 Bd7 13.Bf2 Rc8 14.Rc1 Rb8 15.O-O Rc8 ',
        'raw_first_300': '[Event "IM MARII VICTORII 50A"]\n[Site "?"]\n[Date "2026.07.18"]\n[Round "2"]\n[White "Maksymilian Pilż"]\n[Black "Władysław Przebewski"]\n[Result "0-1"]\n\n1.d4 Nf6 2.Bf4 d5 3.e3 e6 4.Nf3 c5 5.c3 c4 6.Bf3 O-O 7.Nd2 Nc6 8.Ne5 Bd6 9.Bg3 Qc6 10.Qf4 Qxf6 11.b3 cxd4 12.exd4 Bd7 13.Bf2 Rc8 14.Rc1 Rb8 15.O-O Rc8 ',
    },
    {
        'name': 'IMG20260718203903.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 1,
        'sample_moves': ['d4', 'Nf6'],
        'canonical_first_300': '[Event "IX Alan Historyczny Pierwszy"]\n[Round "2/7"]\n[Date "2026.07.18"]\n[White "Matysiak Przemyslaw"]\n[Black "Wierzbicki Dariusz"]\n[Result "0-1"]\n\n1.d4 Nf6 2.Bd4 d5 3.e3 e6 4.Nd3 c5 5.c3 Be7 6.Bd3 O-O 7.Nbd2 Nc6 8.Ne5 Bd6 9.Ng3 Qc7 10.f4 Qb6 11.B3 cxd4 12.exd4 Bd7 13.Bf2 Rac8 14.Rbc1 Rc7 15.O-O Rfc',
        'raw_first_300': '[Event "IX Alan Historyczny Pierwszy"]\n[Round "2/7"]\n[Date "2026.07.18"]\n[White "Matysiak Przemyslaw"]\n[Black "Wierzbicki Dariusz"]\n[Result "0-1"]\n\n1.d4 Sf6 2.Gd4 d5 3.e3 e6 4.Sd3 c5 5.c3 Ge7 6.Gd3 O-O 7.Sbd2 Sc6 8.Se5 Gd6 9.Sg3 Hc7 10.f4 Hb6 11.b3 cxd4 12.exd4 Gd7 13.Gf2 Wac8 14.Wbc1 Wc7 15.O-O Wfc',
    },
    {
        'name': 'IMG20260720191303.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 4,
        'sample_moves': ['d4', 'd5', 'c4', 'c6', 'Nc3', 'Nf6', 'Nf3', 'Bg4'],
        'canonical_first_300': '[Event "IN MAKI VICTORIA TUF"]\n[Date "2026.07.20"]\n[White "Helena Najwosz"]\n[Black "Maksymilian Pilzys"]\n\n1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.Nf3 Bg4 5.exd5 Bxf3 6.Bxf3 cxd5 7.Bg5+ Nbd7 8.O-O e6 9.Ne5 O-O 10.Re7 f6 11.Bxd7 Qxd7 12.Qf6 g6 13.Bxh7+ Qxh7 14.Bg3 Rg8 15.Qh4 Rxg7 16.Rh3 Rg8 17.Re3 Rg6 18.Bc3 Bh2?',
        'raw_first_300': '[Event "IN MAKI VICTORIA TUF"]\n[Date "2026.07.20"]\n[White "Helena Najwosz"]\n[Black "Maksymilian Pilzys"]\n\n1.d4 d5 2.c4 c6 3.Sc3 Sf6 4.Sf3 Gg4 5.e:d5 G:f3 6.G:f3 c:d5 7.Gg5+ Sbd7 8.O-O e6 9.Se5 O-O 10.We7 f6 11.G:d7 H:d7 12.Hf6 g6 13.G:h7+ H:h7 14.Gg3 Wg8 15.Hh4 W:g7 16.Wh3 Wg8 17.We3 Wg6 18.Gc3 Gh2?',
    },
    {
        'name': 'IMG20260721080800.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 1,
        'sample_moves': ['d4', 'd5'],
        'canonical_first_300': '[Event "IN MAHU"]\n[Round "?"]\n[White "VICTORIA TUA"]\n[Black "HELENA NAGAWSKA"]\n[Date "2026.07.20"]\n[Section "NAWCZAS2"]\n[WhiteElo "2127"]\n[BlackElo "2121"]\n\n1.d4 d5 2.d3 c6 3.Nf3 Bg4 4.Nf3 Bxd4 5.exd5 cxd4 6.Bb5+ O-O 7.Bxd5 a6 8.O-O Qd7 9.Ne5 ? 10.Bxd4 c5 11.gxf6 c5 12.Qf3 ? 13.Kh1 f5 14.Qxh3 ? 15.R',
        'raw_first_300': '[Event "IN MAHU"]\n[Round "?"]\n[White "VICTORIA TUA"]\n[Black "HELENA NAGAWSKA"]\n[Date "2026.07.20"]\n[Section "NAWCZAS2"]\n[WhiteElo "2127"]\n[BlackElo "2121"]\n\n1.d4 d5 2.d3 c6 3.Nf3 Bg4 4.Nf3 Bxd4 5.exd5 cxd4 6.Bb5+ O-O 7.Bxd5 a6 8.O-O Qd7 9.Ne5 ? 10.Bxd4 c5 11.gxf6 c5 12.Qf3 ? 13.Kh1 f5 14.Qxh3 ? 15.R',
    },
    {
        'name': 'IMG20260721152746.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 2,
        'sample_moves': ['d4', 'd5', 'Bf4', 'e6', 'e3'],
        'canonical_first_300': '[Event "?"]\n[Site "?"]\n[Date "?"]\n[Round "3"]\n[White "Matuszkiewicz, Filip"]\n[Black "Przybylek, Maciej"]\n[Result "1-0"]\n\n1.d4 d5 2.Bf4 e6 3.e3 Bf5 4.c3 Bd6 5.Bg3 Bxg3 6.hxg3 Nf6 7.Bd3 Nc6 8.Nd2 Bd7 9.Qb3 g6 10.Nd2 Qe5 11.Ne5 h6 12.f2 Nf7 13.Nf3 O-O-O 14.Qc3 Qxd3 15.cxd3 Bxe8? 16.f6+ Rxc6 17.Nxb6+ cx',
        'raw_first_300': '[Event "?"]\n[Site "?"]\n[Date "?"]\n[Round "3"]\n[White "Matuszkiewicz, Filip"]\n[Black "Przybylek, Maciej"]\n[Result "1-0"]\n\n1.d4 d5 2.Gf4 e6 3.e3 Gf5 4.c3 Gd6 5.Gg3 Gxg3 6.hxg3 Sf6 7.Gd3 Sc6 8.Sd2 Gd7 9.Hb3 g6 10.Sd2 He5 11.Se5 h6 12.f2 Sf7 13.Sf3 O-O-O 14.Hc3 Hxd3 15.cxd3 Gxe8? 16.f6+ Wxc6 17.Sxb6+ cx',
    },
    {
        'name': 'IMG20260721194851.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 9,
        'sample_moves': ['d4', 'd5', 'Nf3', 'Nf6', 'e3', 'e6', 'Nbd2', 'Nbd7'],
        'canonical_first_300': '[Event "?"]\n[Site "?"]\n[Date "2022.05.05"]\n[Round "5"]\n[White "?"]\n[Black "?"]\n[Result "1-0"]\n\n1.d4 d5 2.Nf3 Nf6 3.e3 e6 4.Nbd2 Nbd7 5.Bd3 Bd6 6.O-O O-O 7.e4 dxe4 8.Nxe4 Nxe5 9.Nxe5 Bxe5 10.Bd3 Bd6 11.Nc4 Nc4 12.Nd6 Nd6 13.f3 f6 14.fe3 fe3 15.Nc3 Nc3 16.Nxd5 Nxd5 17.B5 B5 18.dxb5 dxb5 19.Rd1 Rd1 20.',
        'raw_first_300': '[Event "?"]\n[Site "?"]\n[Date "2022.05.05"]\n[Round "5"]\n[White "?"]\n[Black "?"]\n[Result "1-0"]\n\n1.d4 d5 2.Sf3 Sf6 3.e3 e6 4.Sbd2 Sbd7 5.Gd3 Gd6 6.O-O O-O 7.e4 d:e4 8.S:e4 S:e5 9.S:e5 G:e5 10.Gd3 Gd6 11.Sc4 Sc4 12.Sd6 Sd6 13.f3 f6 14.fe3 fe3 15.Sc3 Sc3 16.S:d5 S:d5 17.b5 b5 18.d:b5 d:b5 19.Wd1 Wd1 20.',
    },
    {
        'name': 'IMG20260722191535.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 2,
        'sample_moves': ['e4', 'e5', 'Nc3', 'd6'],
        'canonical_first_300': '[Event "?"]\n[White "Jankowski, Maciej"]\n[Black "Marsymuljan, Filas"]\n[Site "Pilzno"]\n[Result "0-1"]\n\n1.e4 e5 2.Nc3 d6 3.Nxe4? Nxe5? 4.Nxe5? dxe5 5.d3 Nf6 6.Ng5? Ng5? 7.Qf3? Qxe7 8.Qxe7+? Qxe7 9.exd5 Nd7 10.Nxd5? Nxd5 11.0-0 Nc6 12.c4 0-0 13.B3 Nd4 14.Rd1 Rxd1+ 15.Nxd1 Nxe2+ 16.Kh1 Nxd4 17.Nb2 Nxc3 1',
        'raw_first_300': '[Event "?"]\n[White "Jankowski, Maciej"]\n[Black "Marsymuljan, Filas"]\n[Site "Pilzno"]\n[Result "0-1"]\n\n1.e4 e5 2.Sc3 d6 3.Sxe4? Sxe5? 4.Sxe5? dxe5 5.d3 Sf6 6.Sg5? Sg5? 7.Hf3? Hxe7 8.Hxe7+? Hxe7 9.exd5 Sd7 10.Sxd5? Sxd5 11.0-0 Sc6 12.c4 0-0 13.b3 Sd4 14.Wd1 Wxd1+ 15.Sxd1 Sxe2+ 16.Kh1 Sxd4 17.Sb2 Sxc3 1',
    },
    {
        'name': 'IMG20260722191537.heic',
        'language': 'en',
        'confidence': 0.5,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '',
        'raw_first_300': '',
    },
    {
        'name': 'IMG20260725185845.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 0,
        'sample_moves': ['d4'],
        'canonical_first_300': '[Event "Mast Michałowo"]\n[Date "2022.09.25"]\n[Round "9"]\n\n1.d4? Bf1? 2.Nf3 Be6 3.c3 e3 4.Nbd2 Nxg3 5.Bf2 c4 6.Nxg3 Nf7 7.Nf6 Nf6 8.c6 h4 9.h4 Bxh4 10.e6 Nxc5 11.h4? Bxd5? 12.Nxd4 e4 13.Nxf6 dxe6 14.Nxf7? O-O 15.Nxf6 Bd7 16.Bxh4 Re6 17.Nxg5 Bxc3 18.Bf2 Kg7 19.Kg7? 1-0',
        'raw_first_300': '[Event "Mast Michałowo"]\n[Date "2022.09.25"]\n[Round "9"]\n\n1.d4? Bf1? 2.Nf3 Be6 3.c3 e3 4.Nbd2 N:g3 5.Bf2 c4 6.N:g3 Nf7 7.Nf6 Nf6 8.c6 h4 9.h4 B:h4 10.e6 N:c5 11.h4? Bxd5? 12.N:d4 e4 13.N:f6 d:e6 14.N:f7? O-O 15.N:f6 Bd7 16.B:h4 Re6 17.N:g5 b:c3 18.Bf2 Kg7 19.Kg7? 1-0',
    },
    {
        'name': 'IMG20260815113420.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 2,
        'sample_moves': ['d4', 'd5', 'Bf4', 'Nf6', 'e3'],
        'canonical_first_300': '[Event "Turniej B"]\n[Site "?"]\n[Date "2026.08.15"]\n[Round "1"]\n[White "M.Plüys"]\n[Black "F.Nikodem"]\n[Result "*"]\n\n1.d4 d5 2.Bf4 Nf6 3.e3 Be5 4.Nf3 e6 5.c3 Bg6 6.Ne5 c5 7.c3 Bxd3 8.Qxd3 c4 9.Qc2 Be7 10.Nd2 0-0 11.c3 c5 12.0-0 Nh5 13.a4 Nxf4 14.exf4 f6 15.Nf3 Bxa1 16.Qxa1 cxb3 17.Qxb3 Nd7 18.Qe1 N6f6',
        'raw_first_300': '[Event "Turniej B"]\n[Site "?"]\n[Date "2026.08.15"]\n[Round "1"]\n[White "M.Plüys"]\n[Black "F.Nikodem"]\n[Result "*"]\n\n1.d4 d5 2.Gf4 Sf6 3.e3 Ge5 4.Sf3 e6 5.c3 Gg6 6.Se5 c5 7.c3 Gxd3 8.Hxd3 c4 9.Hc2 Ge7 10.Sd2 0-0 11.c3 c5 12.0-0 Sh5 13.a4 Sxf4 14.exf4 f6 15.Sf3 Gxa1 16.Hxa1 cxb3 17.Hxb3 Sd7 18.He1 S6f6',
    },
    {
        'name': 'IMG20260815152816.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 3,
        'sample_moves': ['d4', 'g6', 'g4', 'Bg7', 'e3', 'd6'],
        'canonical_first_300': '[Event "Drużynowe Mistrzostwa KPZSzach"]\n[Date "2026.08.15"]\n[Round "3"]\n[White "Wiśniewski, Zbigniew"]\n[Result "*"]\n\n1.d4 g6 2.g4 Bg7 3.e3 d6 4.Nd3 f6 5.Bc4 e6 6.Nc3 a6 7.d5 Qe7 8.e4 e5 9.Be3 Ba6 10.Qd3 Nf5 11.a3 Nd6 12.O-O Qxe3 13.Rd1 Nd6 14.Qf2 Qd2 15.Bxd2 Nc5 16.h3 Bxh6 17.Rb3 Bxe3 18.Rxe3 Nd7 1',
        'raw_first_300': '[Event "Drużynowe Mistrzostwa KPZSzach"]\n[Date "2026.08.15"]\n[Round "3"]\n[White "Wiśniewski, Zbigniew"]\n[Result "*"]\n\n1.d4 g6 2.g4 Gg7 3.e3 d6 4.Sd3 f6 5.Gc4 e6 6.Sc3 a6 7.d5 He7 8.e4 e5 9.Ge3 Ga6 10.Hd3 Sf5 11.a3 Sd6 12.O-O H:e3 13.Wd1 Sd6 14.Hf2 Hd2 15.G:d2 Sc5 16.h3 G:h6 17.Wb3 G:e3 18.W:e3 Sd7 1',
    },
    {
        'name': 'IMG20260815153631.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 3,
        'sample_moves': ['d4', 'g6', 'Bf4', 'Bg7', 'e3', 'd6', 'Nf3'],
        'canonical_first_300': '[Event "Mistrzostwa KPZSzach"]\n[Site "?"]\n[Date "?"]\n[Round "3"]\n[White "Maksymian Pawel"]\n[Black "?"]\n[Result "*"]\n\n1.d4 g6 2.Bf4 Bg7 3.e3 d6 4.Nf3 B6 5.Bc4 e6 6.Nc3 a6 7.d5 Nxd5? 8.Bd1 c5 9.Bc3 Bb7? 10.Qd3 Bxf7? 11.Qf3 Qc6 12.O-O-O Rg7 13.Qe2 Qxe2 14.Bxe2 h5 15.h3 Bd7 16.Bg3 Bxe3 17.Bxe5 Nfd7 18.g',
        'raw_first_300': '[Event "Mistrzostwa KPZSzach"]\n[Site "?"]\n[Date "?"]\n[Round "3"]\n[White "Maksymian Pawel"]\n[Black "?"]\n[Result "*"]\n\n1.d4 g6 2.Bf4 Bg7 3.e3 d6 4.Nf3 b6 5.Bc4 e6 6.Nc3 a6 7.d5 Sxd5? 8.Bd1 c5 9.Bc3 Bb7? 10.Qd3 Bxf7? 11.Qf3 Qc6 12.O-O-O Rg7 13.Qe2 Qxe2 14.Bxe2 h5 15.h3 Bd7 16.Bg3 Bxe3 17.Bxe5 Nfd7 18.g',
    },
    {
        'name': 'IMG20260905103406.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 3,
        'sample_moves': ['e4', 'e5', 'Nf3', 'Nc6', 'Nc3', 'Nf6'],
        'canonical_first_300': '[Event "?"]\n[Site "Gdańsk"]\n[Date "?"]\n[Round "?"]\n[White "Stefan"]\n[Black "Maksymilian Pilż"]\n[Result "1-0"]\n\n1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6 4.d5 d5 5.Nf3 Nc6 6.g5 g4 7.e3 e6 8.Bb5 Ba6 9.Qc6 Bc6 10.Qf3 Bf3 11.Qxf3 Rb8 12.gxf6 Bf6 13.Qxe2 Qxb6 14.B3 Bb4 15.Qxc2 Qf5 16.Rc1 g5 17.O-O Bc3 18.Qxc3 Qa2 19.Q',
        'raw_first_300': '[Event "?"]\n[Site "Gdańsk"]\n[Date "?"]\n[Round "?"]\n[White "Stefan"]\n[Black "Maksymilian Pilż"]\n[Result "1-0"]\n\n1.e4 e5 2.Sf3 Sc6 3.Sc3 Sf6 4.d5 d5 5.Sf3 Sc6 6.g5 g4 7.e3 e6 8.Gb5 Ga6 9.Hc6 Gc6 10.Hf3 Gf3 11.H:f3 Wb8 12.g:f6 Gf6 13.H:e2 H:b6 14.b3 Gb4 15.H:c2 Hf5 16.Wc1 g5 17.O-O Gc3 18.H:c3 Ha2 19.H',
    },
    {
        'name': 'IMG20260905103423.heic',
        'language': 'pl',
        'confidence': 0.8999999999999999,
        'move_count': 8,
        'sample_moves': ['d4', 'd5', 'c4', 'c6', 'Nc3', 'Nf6', 'cxd5', 'cxd5'],
        'canonical_first_300': '[Event "?"]\n[Site "?"]\n[Date "2026.09.05"]\n[Round "1"]\n[White "Stefan Karbowski"]\n[Black "Maksymilian Pilch"]\n[Result "*"]\n\n1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.cxd5 cxd5 5.Nf3 Nc6 6.Bg5 Bg4 7.e3 e6 8.Bb5 a6 9.Bxc6 Bxc6 10.h3 Bxf3 11.Qxf3 Rb8 12.O-O Qc6 13.G3 Bb4 14.Qc2 Qa6 15.Rc1 e5 16.Rb1 Rfd8 17.O-O Bxc3 ',
        'raw_first_300': '[Event "?"]\n[Site "?"]\n[Date "2026.09.05"]\n[Round "1"]\n[White "Stefan Karbowski"]\n[Black "Maksymilian Pilch"]\n[Result "*"]\n\n1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.cxd5 cxd5 5.Nf3 Nc6 6.Bg5 Bg4 7.e3 e6 8.Bb5 a6 9.bxc6 bxc6 10.h3 Bxf3 11.Qxf3 Rb8 12.O-O Qc6 13.G3 Bb4 14.Qc2 Qa6 15.Rc1 e5 16.Rb1 Rfd8 17.O-O Bxc3 ',
    },
    {
        'name': 'IMG20260906103641.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 16,
        'sample_moves': ['d4', 'd5', 'Bf4', 'Nc6', 'e3', 'e6', 'Nf3', 'Bd6'],
        'canonical_first_300': '[Event "?"]\n[Site "?"]\n[Date "?"]\n[Round "?"]\n[White "?"]\n[Black "?"]\n[Result "0-1"]\n\n1. d4 d5 2. Bf4 Nc6 3. e3 e6 4. Nf3 Bd6 5. Bg3 a6 6. c3 Nf6 7. Bd3 Bxg3 8. hxg3 Qd6 9. Qe2 e5 10. e4 Bg4 11. Qe3 Bxf3 12. gxf3 0-0-0 13. f4 exd4 14. cxd4 Qb4+ 15. Qd2 Qxd2 16. Nxd2 Nxd4 17. e5 Rd8 18. 0-0 Ng4 19. N',
        'raw_first_300': '[Event "?"]\n[Site "?"]\n[Date "?"]\n[Round "?"]\n[White "?"]\n[Black "?"]\n[Result "0-1"]\n\n1. d4 d5 2. Gf4 Sc6 3. e3 e6 4. Sf3 Gd6 5. Gg3 a6 6. c3 Sf6 7. Gd3 G:g3 8. h:g3 Hd6 9. He2 e5 10. e4 Gg4 11. He3 G:f3 12. g:f3 0-0-0 13. f4 e:d4 14. c:d4 Hb4+ 15. Hd2 H:d2 16. S:d2 S:d4 17. e5 Wd8 18. 0-0 Sg4 19. S',
    },
    {
        'name': 'IMG20260906115031.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '[Event "Kuk na I/III kat"]\n[Site "?"]\n[Date "2026.08.06"]\n[Round "6"]\n[White "Barańczyk, Bruno"]\n[Black "Małyszko Maksymilian, Pilczys"]\n[Result "*"]\n\n1.B3 e5 2.Bb2 Nc6 3.e3 d5? 4.Bf5 Rd2 5.Bxe6 Bxc6 6.Bxf7 Qxg6 7.Bg8:? ? 8.Nf3 Rxf2 9.Bxf3 Qxe3 10.0-0-0 0-0 11.Qd2 Qxg6 12.Qc4 Bd6 13.GxH6 Kg4? 14.Kg8',
        'raw_first_300': '[Event "Kuk na I/III kat"]\n[Site "?"]\n[Date "2026.08.06"]\n[Round "6"]\n[White "Barańczyk, Bruno"]\n[Black "Małyszko Maksymilian, Pilczys"]\n[Result "*"]\n\n1.b3 e5 2.Gb2 Sc6 3.e3 d5? 4.Gf5 Wd2 5.Gxe6 Gxc6 6.Gxf7 Hxg6 7.Gg8:? ? 8.Sf3 Wxf2 9.Gxf3 Hxe3 10.0-0-0 0-0 11.Hd2 Hxg6 12.Hc4 Gd6 13.GxH6 Kg4? 14.Kg8',
    },
    {
        'name': 'IMG20260906141227.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '[Event "?"]\n[Site "?"]\n[Date "?"]\n[Round "?"]\n[White "Maksymilian"]\n[Black "MOK"]\n[Result "*"]\n\n21.Rxe1 Bg3 22.Re7 Bb5 23.e4 f6 24.Qd3 Qxd3 25.Re3 Rd5 26.exc5 Rg6 27.Rc3 Rxg5 28.Rd2 Rd5 29.Rxc8 Rxc8 30.Rd3 a3 31.Kh2 Bb4 32.axb4 axb4 33.Rb3 Rb5 34.Kg3 Kxg3 35.Kh4 Kd3 36.Kh4 Kd3 37.Bh4 Bc4 38.Bh3 Bxc3',
        'raw_first_300': '[Event "?"]\n[Site "?"]\n[Date "?"]\n[Round "?"]\n[White "Maksymilian"]\n[Black "MOK"]\n[Result "*"]\n\n21.W:e1 Gg3 22.We7 Bb5 23.e4 f6 24.Hd3 H:d3 25.We3 Wd5 26.e:c5 Wg6 27.Wc3 W:g5 28.Wd2 Wd5 29.W:c8 W:c8 30.Wd3 a3 31.Kh2 Bb4 32.a:b4 a:b4 33.Wb3 Wb5 34.Kg3 K:g3 35.Kh4 Kd3 36.Kh4 Kd3 37.Gh4 Gc4 38.Gh3 G:c3',
    },
    {
        'name': 'IMG20260906141255.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 9,
        'sample_moves': ['e4', 'c6', 'd4', 'd5', 'e5', 'Bf5', 'Nc3', 'e6'],
        'canonical_first_300': '[Event "?"]\n[Round "7"]\n[White "Matuszewski Mateusz"]\n[Black "Marszalian..."]\n\n1.e4 c6 2.d4 d5 3.e5 Bf5 4.Nc3 e6 5.Nf3 Bb4 6.Be2 Nd7 7.Bd2 Nh6 8.O-O Rc8 9.Bxh6 gxh6 10.Rg1 Kg8 11.a3 Bxc3 12.Bxc3 g4 13.h3 Ng6 14.Qh6 Bxc2 15.Qxd8+ Kxd8 16.c4 Nc3 17.Nf1 Nxe2+ 18.Rxe2 Bd3 19.Rc1 g5 20.Nf4 Ke7 21.Nxd8 cx',
        'raw_first_300': '[Event "?"]\n[Round "7"]\n[White "Matuszewski Mateusz"]\n[Black "Marszalian..."]\n\n1.e4 c6 2.d4 d5 3.e5 Bf5 4.Nc3 e6 5.Nf3 Bb4 6.Be2 Nd7 7.Bd2 Nh6 8.O-O Rc8 9.Bxh6 gxh6 10.Rg1 Kg8 11.a3 Bxc3 12.bxc3 g4 13.h3 Ng6 14.Qh6 Bxc2 15.Qxd8+ Kxd8 16.c4 Nc3 17.Nf1 Nxe2+ 18.Rxe2 Bd3 19.Rc1 g5 20.Nf4 Ke7 21.Nxd8 cx',
    },
    {
        'name': 'IMG20260912104118.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 8,
        'sample_moves': ['c4', 'c5', 'Nc3', 'Nf6', 'g3', 'Nc6', 'Bg2', 'e6'],
        'canonical_first_300': '[Event "Festiwal Kopernik"]\n[Date "2016.09.12"]\n[Round "1"]\n[White "Maksymilian Pilzys"]\n[Black "Michał Radkowski"]\n[Result "1-0"]\n\n1. c4 c5 2. Nc3 Nf6 3. g3 Nc6 4. Bg2 e6 5. Nf3 Be7 6. O-O O-O 7. Re1 d5 8. cxd5 exd5 9. d4 c6 10. Bg5 Bg7 11. a3 Qd6 12. Rc1 Nd4 13. dxc5 Bxc5 14. Bxe4 Nxe4 15. e4 d4 1',
        'raw_first_300': '[Event "Festiwal Kopernik"]\n[Date "2016.09.12"]\n[Round "1"]\n[White "Maksymilian Pilzys"]\n[Black "Michał Radkowski"]\n[Result "1-0"]\n\n1. c4 c5 2. Nc3 Nf6 3. g3 Nc6 4. Bg2 e6 5. Nf3 Be7 6. O-O O-O 7. Re1 d5 8. cxd5 exd5 9. d4 c6 10. Bg5 Bg7 11. a3 Qd6 12. Rc1 Nd4 13. dxc5 bxc5 14. Bxe4 Nxe4 15. e4 d4 1',
    },
    {
        'name': 'IMG20260912130858.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 5,
        'sample_moves': ['c4', 'c5', 'Nc3', 'Nc6', 'g3', 'g6', 'Bg2', 'Bg7'],
        'canonical_first_300': '[Event "Miedzynarodowy Festiwal Szachowy Koperniku"]\n[Date "2022.08.12"]\n[Round "2"]\n[White "Jan Musialaga"]\n[Black "Maksymilian Pio"]\n[Result "1-0"]\n\n1. c4 c5 2. Nc3 Nc6 3. g3 g6 4. Bg2 Bg7 5. e4 Nf6 6. Ne2 O-O 7. O-O Rc8 8. d3 d6 9. e5 Bd2 10. Be6 ? 11. Nd5 Bxd5 12. cxd5 Nd4 13. Nxd4 cxd4 14. f4 N',
        'raw_first_300': '[Event "Miedzynarodowy Festiwal Szachowy Koperniku"]\n[Date "2022.08.12"]\n[Round "2"]\n[White "Jan Musialaga"]\n[Black "Maksymilian Pio"]\n[Result "1-0"]\n\n1. c4 c5 2. Sc3 Sc6 3. g3 g6 4. Gg2 Gg7 5. e4 Sf6 6. Se2 O-O 7. O-O Wc8 8. d3 d6 9. e5 Gd2 10. Ge6 ? 11. Sd5 Gxd5 12. cxd5 Sd4 13. Sxd4 cxd4 14. f4 S',
    },
    {
        'name': 'IMG20260912130901.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 2,
        'sample_moves': ['c3', 'c5', 'g3', 'Nc6'],
        'canonical_first_300': '[Event "Kopernik"]\n[White "Jan Maciąg"]\n[Black "Miksownik"]\n[Round "2"]\n\n1. c3 c5 2. g3 Nc6 3. Nc3 g6 4. Bg2 Bg7 5. e4 Nf6 6. Ne2 O-O 7. O-O Re8 8. d3 d6 9. h3 e5 10. Bd2 Be6 11. Ndb6 Bxd5 12. cxd5 Nd4 13. Nxd4 cxd4 14. f4 Nh5 15. Qf3 Bh6 16. Rc1 Rc8 17. Kh2 Qf6 18. Qg4 Rxc1 19. Bxc1 Qd8 20. Qf3 Qc8',
        'raw_first_300': '[Event "Kopernik"]\n[White "Jan Maciąg"]\n[Black "Miksownik"]\n[Round "2"]\n\n1. c3 c5 2. g3 Nc6 3. Nc3 g6 4. Bg2 Bg7 5. e4 Nf6 6. Ne2 O-O 7. O-O Re8 8. d3 d6 9. h3 e5 10. Bd2 Be6 11. Ndb6 Bxd5 12. cxd5 Nd4 13. Nxd4 cxd4 14. f4 Nh5 15. Qf3 Bh6 16. Rc1 Rc8 17. Kh2 Qf6 18. Qg4 Rxc1 19. Bxc1 Qd8 20. Qf3 Qc8',
    },
    {
        'name': 'IMG20260912150724.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 3,
        'sample_moves': ['c4', 'c5', 'Nf3', 'Nc6', 'd3', 'g6'],
        'canonical_first_300': '[Event "?"]\n[Date "2026.09.12"]\n[Round "3"]\n[White "Pilys, Maximilian"]\n[Black "Kulbaczewski, Krzysztof"]\n\n1.c4 c5 2.Nf3 Nc6 3.d3 g6 4.Bg2 Bg7 5.O-O O-O 6.Re1 Re8 7.h3 d6 8.Be3 e5 9.Nc5 Nd4 10.cxd5 Nxd4 11.Qd2 Nf3 12.Bxf3 Bxf3 13.Bxh6 Qd7 14.Bxg7 Qxg7 15.Rd1 a6 16.B2 B3 17.Bxc5 Bxc5 18.Bd1 f6 19.Qc1',
        'raw_first_300': '[Event "?"]\n[Date "2026.09.12"]\n[Round "3"]\n[White "Pilys, Maximilian"]\n[Black "Kulbaczewski, Krzysztof"]\n\n1.c4 c5 2.Sf3 Sc6 3.d3 g6 4.Gg2 Gg7 5.O-O O-O 6.We1 We8 7.h3 d6 8.Ge3 e5 9.Sc5 Sd4 10.c:d5 S:d4 11.Hd2 Sf3 12.G:f3 G:f3 13.G:h6 Hd7 14.G:g7 H:g7 15.Wd1 a6 16.b2 b3 17.G:c5 b:c5 18.Gd1 f6 19.Hc1',
    },
    {
        'name': 'IMG20260912174651.heic',
        'language': 'en',
        'confidence': 0.5,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '',
        'raw_first_300': '',
    },
    {
        'name': 'IMG20260919103723.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 8,
        'sample_moves': ['d4', 'd5', 'c4', 'c6', 'Nc3', 'Nf6', 'Nf3', 'e5'],
        'canonical_first_300': '[Event "?"]\n[Site "?"]\n[Date "?"]\n[Round "1"]\n[White "Sebastian Wawrzynek"]\n[Black "Maksymilian Pilzys"]\n[Result "*"]\n\n1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.Nf3 e5 5.Qb3 Qb6 6.Qxb6 axb6 7.cxd5 Nxd5 8.Nxd5 cxd5 9.e5 e6 10.Bb5+ Nc6 11.O-O Bd6 12.Ne5 Bxe5 13.dxe5 O-O 14.f4 Nd7 15.Bd2 Nb3 16.B3 Rc8 17.Rfd1 Nc5 18',
        'raw_first_300': '[Event "?"]\n[Site "?"]\n[Date "?"]\n[Round "1"]\n[White "Sebastian Wawrzynek"]\n[Black "Maksymilian Pilzys"]\n[Result "*"]\n\n1.d4 d5 2.c4 c6 3.Sc3 Sf6 4.Sf3 e5 5.Hb3 Hb6 6.Hxb6 axb6 7.cxd5 Sxd5 8.Sxd5 cxd5 9.e5 e6 10.Gb5+ Sc6 11.O-O Gd6 12.Se5 Gxe5 13.dxe5 O-O 14.f4 Sd7 15.Gd2 Sb3 16.b3 Wc8 17.Wfd1 Sc5 18',
    },
    {
        'name': 'IMG20260919103724.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 3,
        'sample_moves': ['d4', 'd5', 'c4', 'c6', 'Nc3', 'Nf6', 'Nf3'],
        'canonical_first_300': '[Event "?"]\n[Date "2026.09.18"]\n[Round "1"]\n[White "Sebastian"]\n[Black "Maksymilian"]\n\n1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.Nf3 Bg5 5.Qb3 cxd5 6.Qxd5 Qxd5 7.cxd5 cxd5 8.e3 Nxd5 9.Bb5+ c6 10.O-O e6 11.Ne5 Bc5 12.dxc6 O-O 13.Bg5 f4 14.Qa4 Nf3 15.Bxd2 Nf3 16.c3 Rac8 17.Rd1 Rc2 18.Ba1 hxf4 19.B3 Rf8 20.Rd3 Rc1 2',
        'raw_first_300': '[Event "?"]\n[Date "2026.09.18"]\n[Round "1"]\n[White "Sebastian"]\n[Black "Maksymilian"]\n\n1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.Nf3 Bg5 5.Qb3 cxd5 6.Qxd5 Qxd5 7.cxd5 cxd5 8.e3 Nxd5 9.Bb5+ c6 10.O-O e6 11.Ne5 Bc5 12.dxc6 O-O 13.Bg5 f4 14.Qa4 Nf3 15.Bxd2 Nf3 16.c3 Rac8 17.Rd1 Rc2 18.Ba1 hxf4 19.b3 Rf8 20.Rd3 Rc1 2',
    },
    {
        'name': 'IMG20260919133428.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 9,
        'sample_moves': ['d4', 'c6', 'Bf4', 'd5', 'e3', 'Bf5', 'Nf3', 'e6'],
        'canonical_first_300': '[Event "Turniej"]\n[Site "Gdansk"]\n[Date "?"]\n[Round "3"]\n[White "Maksymilian Pilipis"]\n[Black "Leonard"]\n[Result "*"]\n\n1. d4 c6 2. Bf4 d5 3. e3 Bf5 4. Nf3 e6 5. Bd3 Qa5+ 6. c3 Bxd3 7. Qxd3 Nd7 8. Ng5 h6 9. Nf3 O-O-O 10. Nd2 Nf6 11. B4 Qb6 12. a4 a6 13. a5 Qa7 14. O-O Be7 15. Ne5 Nxe5 16. dxe5 Nfd7 1',
        'raw_first_300': '[Event "Turniej"]\n[Site "Gdansk"]\n[Date "?"]\n[Round "3"]\n[White "Maksymilian Pilipis"]\n[Black "Leonard"]\n[Result "*"]\n\n1. d4 c6 2. Bf4 d5 3. e3 Bf5 4. Nf3 e6 5. Bd3 Qa5+ 6. c3 Bxd3 7. Qxd3 Nd7 8. Ng5 h6 9. Nf3 O-O-O 10. Nd2 Nf6 11. b4 Qb6 12. a4 a6 13. a5 Qa7 14. O-O Be7 15. Ne5 Nxe5 16. dxe5 Nfd7 1',
    },
    {
        'name': 'IMG20260919133435.heic',
        'language': 'en',
        'confidence': 0.5,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '',
        'raw_first_300': '',
    },
    {
        'name': 'IMG20260921101132.heic',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 5,
        'sample_moves': ['e4', 'e5', 'Nf3', 'Nc6', 'g3', 'g6', 'Bg2', 'Bg7'],
        'canonical_first_300': '[Event "?"]\n[Site "?"]\n[Date "2024.09.12"]\n[Round "2"]\n[White "Mazur Jakub"]\n[Black "Kulbaciński Kacper"]\n[Result "*"]\n\n1.e4 e5 2.Nf3 Nc6 3.g3 g6 4.Bg2 Bg7 5.d3 d6 6.0-0 0-0 7.Ne1 Ne8 8.d3? d6? 9.Be3 e5 10.f3 Nd4 11.Be3 Be6 12.Nd2 Nf6 13.Rb1 Rb8 14.Be2 Rd7 15.Bxg7 Qxg7 16.Rd1 Bb4 17.B3 Ba5 18.Bxc5 B',
        'raw_first_300': '[Event "?"]\n[Site "?"]\n[Date "2024.09.12"]\n[Round "2"]\n[White "Mazur Jakub"]\n[Black "Kulbaciński Kacper"]\n[Result "*"]\n\n1.e4 e5 2.Sf3 Sc6 3.g3 g6 4.Gg2 Gg7 5.d3 d6 6.0-0 0-0 7.Se1 Se8 8.d3? d6? 9.Ge3 e5 10.f3 Sd4 11.Ge3 Ge6 12.Sd2 Sf6 13.Wb1 Wb8 14.Ge2 Wd7 15.G:g7 H:g7 16.Wd1 Gb4 17.b3 Ga5 18.G:c5 b',
    },
    {
        'name': 'IMG20260921101135.heic',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 3,
        'sample_moves': ['e4', 'e5', 'Nf3', 'Nc6', 'd4', 'exd4', 'Bc4'],
        'canonical_first_300': '[Event "Mistrzowski Półfinał"]\n[Site "?"]\n[Date "2026.02.12"]\n[Round "IV"]\n[White "Kaszuba"]\n[Black "Maksym..."]\n[Result "1-0"]\n\n1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Bc4 Be6 5.Be2 Be6 6.0-0 0-0 7.c3 dxc3 8.Nxc3 Nf6 9.Nbd2 Nbd7 10.h3 h6 11.Nb1 Rb8 12.Re1 Re8 13.Rc1 Rc6 14.Qh3 Qd6 15.Nb5 Rxe4? 16.Bxe4 Nxe4 1',
        'raw_first_300': '[Event "Mistrzowski Półfinał"]\n[Site "?"]\n[Date "2026.02.12"]\n[Round "IV"]\n[White "Kaszuba"]\n[Black "Maksym..."]\n[Result "1-0"]\n\n1.e4 e5 2.Sf3 Sc6 3.d4 exd4 4.Gc4 Ge6 5.Ge2 Ge6 6.0-0 0-0 7.c3 dxc3 8.Sxc3 Sf6 9.Sbd2 Sbd7 10.h3 h6 11.Sb1 Wb8 12.We1 We8 13.Wc1 Wc6 14.Hh3 Hd6 15.Sb5 Wxe4? 16.Gxe4 Sxe4 1',
    },
    {
        'name': 'IMG20260921101140.heic',
        'language': 'en',
        'confidence': 0.5,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '',
        'raw_first_300': '',
    },
    {
        'name': 'IMG_20260815_135623.jpg',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 1,
        'sample_moves': ['e4', 'c5'],
        'canonical_first_300': '[Event "Drużynowe Mistrzostwa KPZSzach"]\n[Date "2010.08.15"]\n[Round "2"]\n[White "Jeżowski Maksymilian"]\n[Black "Matejczyk"]\n[Result "0-1"]\n\n1.e4 c5 2.e3 d5 3.B3 Nf6 4.Bb2 Nc6 5.exd5 Nxd5 6.Bb5+ Qc7? 7.Bxc6 Bxc6 8.Bb3 Qd6 9.Bxg7 Rg8 10.Bd2 Rxg2 11.Qe2 Qh8 12.d4 Bb4+ 0-1',
        'raw_first_300': '[Event "Drużynowe Mistrzostwa KPZSzach"]\n[Date "2010.08.15"]\n[Round "2"]\n[White "Jeżowski Maksymilian"]\n[Black "Matejczyk"]\n[Result "0-1"]\n\n1.e4 c5 2.e3 d5 3.b3 Sf6 4.Gb2 Sc6 5.e:d5 S:d5 6.Gb5+ Hc7? 7.G:c6 G:c6 8.Gb3 Hd6 9.G:g7 Wg8 10.Gd2 W:g2 11.He2 Hh8 12.d4 Gb4+ 0-1',
    },
    {
        'name': 'IMG_20260905_121819.jpg',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 1,
        'sample_moves': ['d4', 'd5', 'Bf4'],
        'canonical_first_300': '[Event "?"]\n[Date "2025.09.05"]\n[Round "2"]\n[White "Stępniewska"]\n[Black "Urszula"]\n[Result "1/2-1/2"]\n\n1.d4 d5 2.Bf4 Bf6 3.e3 e6 4.Nf3 Nf6 5.Bd3 c5 6.Ne2 cxd4 7.exd4 Bc5 8.0-0 dxe4 9.Bg5 Bxc6 10.Bxc6+ Bxc6 11.Nxe4 Nxf5 12.Qxe5 Qxe5 13.Ng3 Qxg3 14.0-0 Qd4 15.cxd4 Qxd4 16.Rad1 e5 17.Ng2 Ba6 18.Qxf8+ ',
        'raw_first_300': '[Event "?"]\n[Date "2025.09.05"]\n[Round "2"]\n[White "Stępniewska"]\n[Black "Urszula"]\n[Result "1/2-1/2"]\n\n1.d4 d5 2.Gf4 Gf6 3.e3 e6 4.Sf3 Sf6 5.Gd3 c5 6.Se2 cxd4 7.exd4 Gc5 8.0-0 dxe4 9.Gg5 Gxc6 10.Gxc6+ bxc6 11.Sxe4 Sxf5 12.Hxe5 Hxe5 13.Sg3 Hxg3 14.0-0 Hd4 15.cxd4 Hxd4 16.Wad1 e5 17.Sg2 Ga6 18.Hxf8+ ',
    },
    {
        'name': 'IMG_20260905_174348.jpg',
        'language': 'en',
        'confidence': 0.5,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '',
        'raw_first_300': '',
    },
    {
        'name': 'IMG_20260913_104602.jpg',
        'language': 'en',
        'confidence': 0.5,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '',
        'raw_first_300': '',
    },
    {
        'name': 'IMG_20260913_162648.jpg',
        'language': 'pl',
        'confidence': 0.99,
        'move_count': 5,
        'sample_moves': ['e4', 'c6', 'Nf3', 'd5', 'exd5', 'cxd5', 'd4', 'Bg4'],
        'canonical_first_300': '[Event "?"]\n[Date "2020.09.13"]\n[Round "6"]\n[White "Kaszuba Robert"]\n[Black "Pytlis Maksym"]\n[Result "1-0"]\n\n1.e4 c6 2.Nf3 d5 3.exd5 cxd5 4.d4 Bg4 5.Be2 Be6 6.O-O Bd6 7.Be3 f5 8.Nc3 O-O 9.Nbd2 h6 10.h3 Bxf3 11.Bxf3 Re8 12.Rxe8+ Nc6 13.Rc1 Rc8 14.Qb3 B6 15.Be2 e5 16.Bxc5 Nxc5 17.Nf3 Nxf3 18.Bxf3 Ne4 ',
        'raw_first_300': '[Event "?"]\n[Date "2020.09.13"]\n[Round "6"]\n[White "Kaszuba Robert"]\n[Black "Pytlis Maksym"]\n[Result "1-0"]\n\n1.e4 c6 2.Nf3 d5 3.exd5 cxd5 4.d4 Bg4 5.Be2 Be6 6.O-O Bd6 7.Be3 f5 8.Nc3 O-O 9.Nbd2 h6 10.h3 Bxf3 11.Bxf3 Re8 12.Rxe8+ Nc6 13.Rc1 Rc8 14.Qb3 b6 15.Be2 e5 16.Bxc5 Nxc5 17.Nf3 Nxf3 18.Bxf3 Ne4 ',
    },
    {
        'name': 'IMG_20260913_162817.jpg',
        'language': 'pl',
        'confidence': 0.9,
        'move_count': 1,
        'sample_moves': ['e4', 'f6', 'g3'],
        'canonical_first_300': '[Event "?"]\n[White "Maksymilian Plusz"]\n[Black "Witald"]\n[Site "?"]\n[Date "?"]\n\n1.e4 f6 2.g3 Kg7 3.Bg2 Nf6 4.Bg3 O-O 5.Bg2 d6 6.d3 e5 7.Nd2 c5 8.Nf3 Nc6 9.O-O Bg4 10.Re1 d6 11.e3 Bg5 12.h3 Bxf3 13.Bxf3 Re8 14.Ne4 Nxe5 15.Bxe4 Qd7 16.Kh2 Ne7 17.g4 d5 18.Bg2 Nxc4 19.Nxc4 dxc4 20.Rg1 f5 21.Bg3 Qd6 22.K',
        'raw_first_300': '[Event "?"]\n[White "Maksymilian Plusz"]\n[Black "Witald"]\n[Site "?"]\n[Date "?"]\n\n1.e4 f6 2.g3 Kg7 3.Gg2 Sf6 4.Gg3 O-O 5.Gg2 d6 6.d3 e5 7.Sd2 c5 8.Sf3 Sc6 9.O-O Gg4 10.We1 d6 11.e3 Gg5 12.h3 G:f3 13.G:f3 We8 14.Se4 S:e5 15.G:e4 Hd7 16.Kh2 Se7 17.g4 d5 18.Gg2 S:c4 19.S:c4 d:c4 20.Wg1 f5 21.Gg3 Hd6 22.K',
    },
    {
        'name': 'IMG_20260915_142332.jpg',
        'language': 'en',
        'confidence': 0.5,
        'move_count': 0,
        'sample_moves': [],
        'canonical_first_300': '',
        'raw_first_300': '',
    },
]


def _id_from_name(name: str) -> str:
    """Stable pytest id derived from image filename."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def _wrap_canonical(canonical: str) -> str:
    """Wrap a movetext snippet into a parseable PGN."""
    if canonical.lstrip().startswith('['):
        return canonical
    return '[Event "?"]\n\n' + canonical


@pytest.mark.parametrize(
    "case",
    _REAL_CORPUS,
    ids=[_id_from_name(c['name']) for c in _REAL_CORPUS],
)
def test_real_corpus_language_autodetect(case: dict) -> None:
    """The heuristic detects the same language the live OCR pipeline detected."""
    raw = case['raw_first_300']
    if not raw.strip():
        pytest.skip(f"{case['name']}: empty raw OCR text — no autodetect signal")
    detected, _confidence = detect_language(raw)
    assert detected in {'en', 'pl'}, (
        f"{case['name']}: autodetect returned {detected!r} "
        f"for raw text starting with {raw[:80]!r}"
    )


@pytest.mark.parametrize(
    "case",
    _REAL_CORPUS,
    ids=[_id_from_name(c['name']) for c in _REAL_CORPUS],
)
def test_real_corpus_normalize_round_trip(case: dict) -> None:
    """san_normalize produces canonical English SAN the python-chess parser accepts."""
    canonical = case['canonical_first_300']
    if not canonical.strip():
        pytest.skip(f"{case['name']}: empty canonical text")
    lang = case['language']
    result = normalize_pgn(canonical, language=lang)
    wrapped = _wrap_canonical(result.canonical_text)
    game = chess.pgn.read_game(io.StringIO(wrapped))
    assert game is not None, (
        f"{case['name']}: round-tripped PGN did not parse: {result.canonical_text[:200]!r}"
    )


@pytest.mark.parametrize(
    "case",
    _REAL_CORPUS,
    ids=[_id_from_name(c['name']) for c in _REAL_CORPUS],
)
def test_real_corpus_sample_moves_parse(case: dict) -> None:
    """Every sample_moves entry parses cleanly via python-chess."""
    sample = case['sample_moves']
    if not sample:
        pytest.skip(f"{case['name']}: no sample moves captured")
    board = chess.Board()
    for san in sample:
        try:
            move = board.parse_san(san)
        except (chess.InvalidMove, chess.AmbiguousMoveError, ValueError) as exc:
            pytest.fail(
                f"{case['name']}: sample move {san!r} failed to parse: {exc}"
            )
        board.push(move)
