#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unittests for the makedep.py script
"""

from __future__ import absolute_import

import unittest
import tempfile
import shutil
import json
import io
import os
from os import path

import makedep


class TestCheckArchives(unittest.TestCase):
    """Test case for the main method in makedep"""

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base_dir)

    def test_empty_dir(self):
        """
        running on an empty dir with no files generates an empty deps file
        """
        my_dir = path.join(self.base_dir, "empty")
        out_fn = path.join(my_dir, "all.dep")

        os.mkdir(my_dir)

        makedep.main(
            out_fn,
            "lower", "normal", ".a",
            my_dir, [])

        with open(out_fn, 'r') as fhandle:
            no_comment_lines = [l.strip() for l in fhandle if not l.startswith('#') and l.strip()]

        self.assertEqual(len(no_comment_lines), 0)

    def test_single_empty(self):
        """
        running on a dir with a single file
        """

        my_dir = path.join(self.base_dir, "single_empty")
        out_fn = path.join(my_dir, "all.dep")
        single_fn = path.join(my_dir, "single.F")
        pkg_fn = path.join(my_dir, "PACKAGE")

        os.mkdir(my_dir)
        with open(single_fn, 'w') as fhandle:
            fhandle.write("\n")

        with self.assertRaises(SystemExit):
            # should throw an exception due to missing PACKAGES
            makedep.main(
                out_fn,
                "lower", "normal", ".a",
                my_dir, ["./single.F"])

        with open(pkg_fn, 'w') as fhandle:
            json.dump({
                "description": "Nothing",
                "archive": "test",
                "public": ["*.F"],
                "requires": []
                }, fhandle)

        makedep.main(
            out_fn,
            "lower", "normal", ".a",
            my_dir, ["./single.F"])

        with open(out_fn, 'r') as fhandle:
            no_comment_lines = [l.strip() for l in fhandle if not l.startswith('#') and l.strip()]

        self.assertEqual(
            no_comment_lines,
            ['$(LIBDIR)/test.a : single.o', 'install: PUBLICFILES += *.F', 'single.o : single.F'])


    def test_unicode(self):
        """
        running on a dir with a single file
        """

        my_dir = path.join(self.base_dir, "unicode")
        out_fn = path.join(my_dir, "all.dep")
        single_fn = path.join(my_dir, "single.F")
        pkg_fn = path.join(my_dir, "PACKAGE")

        os.mkdir(my_dir)
        with io.open(single_fn, 'w', encoding='utf8') as fhandle:
            fhandle.write(u"! Ångström\n")

        with io.open(pkg_fn, 'w', encoding='utf8') as fhandle:
            fhandle.write(u"""{
    "description": "unicode test with just an Ångström 😉",
    "archive": "test",
    "public": ["*.F"],
    "requires": []
}""")

        makedep.main(
            out_fn,
            "lower", "normal", ".a",
            my_dir, ["./single.F"])

        with open(out_fn, 'r') as fhandle:
            no_comment_lines = [l.strip() for l in fhandle if not l.startswith('#') and l.strip()]

        self.assertEqual(
            no_comment_lines,
            ['$(LIBDIR)/test.a : single.o', 'install: PUBLICFILES += *.F', 'single.o : single.F'])


if __name__ == '__main__':
    unittest.main()
