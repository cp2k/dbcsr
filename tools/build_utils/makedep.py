#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import ast
from os import path
from os.path import dirname, basename, normpath
import glob

# pre-compiled regular expressions
re_module = re.compile(r"(?:^|\n)\s*module\s+(\w+)\s.*\n\s*end\s*module", re.DOTALL)
re_program = re.compile(r"\n\s*end\s*program")
re_main = re.compile(r"\sint\s+main\s*\(")
re_use = re.compile(r"\n\s*use\s+(\w+)")
re_incl_fypp = re.compile(r"\n#:include\s+['\"](.+)['\"]")
re_incl_cpp = re.compile(r"\n#include\s+['\"](.+)['\"]")
re_incl_fort = re.compile(r"\n\s*include\s+['\"](.+)['\"]")


# ============================================================================
def main(out_fn, project_name, mod_format, mode, archive_ext, src_dir, src_files):
    messages = []
    # process command line arguments
    src_files = [path.join(src_dir, f) for f in src_files]

    if mod_format not in ('lower', 'upper', 'no'):
        error('Module filename format must be eighter of "lower", "upper", or "no".')

    if mode not in ('normal', 'hackdep', 'mod_compiler'):
        error('Mode must be eighter of "normal", "hackdep", or "mod_compiler".')

    for fn in src_files:
        if not fn.startswith("/"):
            error("Path of source-file not absolut: "+fn)

    src_basenames = [basename(fn).rsplit(".", 1)[0] for fn in src_files]
    for bfn in src_basenames:
        if src_basenames.count(bfn) > 1:
            error("Multiple source files with the same basename: "+bfn)

    # parse files
    parsed_files = dict()
    for fn in src_files:
        parse_file(parsed_files, fn, src_dir) #parses also included files
    messages.append("Parsed %d files" % len(parsed_files))

    # create table mapping fortan module-names to file-name
    mod2fn = dict()
    for fn in src_files:
        for m in parsed_files[fn]['module']:
            if m in mod2fn.keys():
                error('Multiple declarations of module "%s"'%m)
            mod2fn[m] = fn
    messages.append("Created mod2fn table, found %d modules." % len(mod2fn))

    # check "one module per file"-convention
    for m, fn in mod2fn.items():
        if basename(fn) != m+".F":
            error("Names of module and file do not match: "+fn)

    # read package manifests
    packages = dict()
    for fn in src_files:
        p = normpath(dirname(fn))
        read_pkg_manifest(project_name, packages, p)
    messages.append("Read %d package manifests" % len(packages))

    # check dependencies against package manifests
    n_deps = 0
    for fn in src_files:
        p = normpath(dirname(fn))
        if not parsed_files[fn]['program']:
            packages[p]['objects'].append(src2obj(basename(fn)))
        deps = collect_include_deps(parsed_files, fn, src_dir)
        deps += [mod2fn[m] for m in collect_use_deps(parsed_files, fn, src_dir) if m in mod2fn.keys()]
        n_deps += len(deps)
        for d in deps:
            dp = normpath(dirname(d))
            if dp not in packages[p]['allowed_deps']:
                error("Dependency forbidden according to package manifest: %s -> %s"%(fn, d))
            if dp != p and "public_files" in packages[dp].keys():
                if basename(d) not in packages[dp]["public_files"]:
                    error("File not public according to package manifest: %s -> %s"%(fn, d))
    messages.append("Checked %d dependencies"%n_deps)

    # check for circular dependencies
    for fn in parsed_files.keys():
        find_cycles(parsed_files, mod2fn, fn, src_dir)

    # write messages as comments
    makefile = "".join(["#makedep: %s\n"%m for m in messages])
    makefile += "\n"

    # write rules for archives
    for p in packages.keys():
        if packages[p]['objects']:
            makefile += "# Package %s\n"%p
            makefile += "$(LIBDIR)/%s : "%(packages[p]['archive']+archive_ext)
            makefile += " ".join(packages[p]['objects']) + "\n\n"

    # write rules for public files
    for p in packages.keys():
        if "public" in packages[p].keys():
            makefile += "# Public modules for package %s\n" % p
            makefile += "install: PUBLICFILES += "
            for mod in packages[p]["public"]:
                makefile += "%s " % mod
            makefile += "\n\n"

    # write rules for executables
    archive_postfix = archive_ext.rsplit(".", 1)[0]
    for fn in src_files:
        if not parsed_files[fn]['program']:
            continue

        bfn = basename(fn).rsplit(".", 1)[0]
        makefile += "# Program %s\n"%fn
        makefile += "$(EXEDIR)/%s.$(ONEVERSION) : %s.o "%(bfn, bfn)
        p = normpath(dirname(fn))
        deps = collect_pkg_deps(packages, p)
        makefile += " ".join(["$(LIBDIR)/"+a+archive_ext for a in deps]) + "\n"
        makefile += "\t" + "$(LD) $(LDFLAGS)"
        if fn.endswith(".c") or fn.endswith(".cu"):
            makefile += " $(LDFLAGS_C)"
        makefile += " -L$(LIBDIR) -o $@ %s.o "%bfn
        makefile += "$(EXTERNAL_OBJECTS) "
        assert all([a.startswith("lib") for a in deps])
        makefile += " ".join(["-l"+a[3:]+archive_postfix for a in deps])
        makefile += " $(LIBS)\n\n"

    # write rules for objects
    for fn in src_files:
        deps = " ".join(collect_include_deps(parsed_files, fn, src_dir))
        mods = collect_use_deps(parsed_files, fn, src_dir)
        mods.sort(key=cmp_mods) # sort mods to speedup compilation
        for m in mods:
            if m in mod2fn.keys():
                deps += " " + mod2modfile(m, mod_format)
        if mode == "hackdep":
            deps = ""

        bfn = basename(fn)
        makefile += "# Object %s\n"%bfn
        provides = [mod2modfile(m, mod_format) for m in parsed_files[fn]['module']]
        for mfn in provides:
            makefile += "%s : %s "%(mfn, bfn) + deps + "\n"
        makefile += "%s : %s "%(src2obj(bfn), bfn) + deps
        if mode == "mod_compiler":
            makefile += " " + " ".join(provides)
        makefile += "\n\n"

    with open(out_fn, "w") as fhandle:
        fhandle.write(makefile)
        fhandle.close()


# ============================================================================
def cmp_mods(mod):
    # list "type" modules first, they are probably on the critical path
    if "type" in mod:
        return 0
    return 1


# ============================================================================
def parse_file(parsed_files, fn, src_dir):
    if fn in parsed_files:
        return

    with open(fn) as fhandle:
        content = fhandle.read()

    # re.IGNORECASE is horribly expensive. Converting to lower-case upfront
    content_lower = content.lower()

    # all files are parsed for cpp includes
    incls = re_incl_cpp.findall(content) #CPP includes (case-sensitiv)

    mods = []
    uses = []
    prog = False

    if fn[-2:] == ".F" or fn[-4:] == ".f90" or fn[-5:] == ".fypp":
        mods += re_module.findall(content_lower)
        prog = re_program.search(content_lower) != None
        uses += re_use.findall(content_lower)
        incls += re_incl_fypp.findall(content) # Fypp includes (case-sensitiv)
        incl_fort_iter = re_incl_fort.finditer(content_lower) # fortran includes
        incls += [content[m.start(1):m.end(1)] for m in incl_fort_iter]

    if fn[-2:] == ".c" or fn[-3:] == ".cu":
        prog = re_main.search(content) != None # C is case-sensitiv

    # exclude included files from outside the source tree
    def incl_fn(i):
        return normpath(path.join(dirname(fn), i))

    def incl_fn_src(i):
        return normpath(path.join(src_dir, i))

    existing_incl = [i for i in incls if path.exists(incl_fn(i))]
    existing_incl_src = [i for i in incls if path.exists(incl_fn_src(i))]

    # store everything in parsed_files cache
    parsed_files[fn] = {'module':mods, 'program': prog, 'use':uses,
                        'include':existing_incl,
                        'include_src':existing_incl_src}

    # parse included files
    for i in existing_incl:
        parse_file(parsed_files, incl_fn(i), src_dir)

    for i in existing_incl_src:
        parse_file(parsed_files, incl_fn_src(i), src_dir)


# ============================================================================
def read_pkg_manifest(project_name, packages, p):
    if p in packages.keys():
        return

    fn = path.join(p, "PACKAGE")
    if not path.exists(fn):
        error("Could not open PACKAGE manifest: " + fn)

    with open(fn) as fhandle:
        content = fhandle.read()

    packages[p] = ast.literal_eval(content)
    packages[p]['objects'] = []
    if "archive" not in packages[p].keys():
        packages[p]['archive'] = "lib{}{}".format(project_name, basename(p))
    packages[p]['allowed_deps'] = [normpath(p)]
    packages[p]['allowed_deps'] += [normpath(path.join(p, r)) for r in packages[p]['requires']]

    for r in packages[p]['requires']:
        read_pkg_manifest(project_name, packages, normpath(path.join(p, r)))

    if "public" in packages[p].keys():
        public_files = []
        for fn in packages[p]["public"]:
            public_files += glob.glob(path.join(p, fn))
        packages[p]["public_files"] = [basename(fn) for fn in public_files]

# ============================================================================
def mod2modfile(m, mod_format):
    if mod_format == 'no':
        return ""

    if mod_format == 'lower':
        return m.lower() + ".mod"

    if mod_format == 'upper':
        return m.upper() + ".mod"

    assert False  # modeps unknown


# ============================================================================
def src2obj(src_fn):
    return basename(src_fn).rsplit(".", 1)[0] + ".o"


# ============================================================================
def collect_include_deps(parsed_files, fn, src_dir):
    pf = parsed_files[fn]
    incs = []

    for i in pf['include']:
        fn_inc = normpath(path.join(dirname(fn), i))
        if fn_inc in parsed_files.keys():
            incs.append(fn_inc)
            incs += collect_include_deps(parsed_files, fn_inc, src_dir)

    for i in pf['include_src']:
        fn_inc = normpath(path.join(src_dir, i))
        if fn_inc in parsed_files.keys():
            incs.append(fn_inc)
            incs += collect_include_deps(parsed_files, fn_inc, src_dir)

    return list(set(incs))


# ============================================================================
def collect_use_deps(parsed_files, fn, src_dir):
    pf = parsed_files[fn]
    uses = pf['use']

    for i in pf['include']:
        fn_inc = normpath(path.join(dirname(fn), i))
        if fn_inc in parsed_files.keys():
            uses += collect_use_deps(parsed_files, fn_inc, src_dir)

    for i in pf['include_src']:
        fn_inc = normpath(path.join(src_dir, i))
        if fn_inc in parsed_files.keys():
            uses += collect_use_deps(parsed_files, fn_inc, src_dir)

    return list(set(uses))


# ============================================================================
def find_cycles(parsed_files, mod2fn, fn, src_dir, S=None):
    pf = parsed_files[fn]
    if 'visited' in pf.keys():
        return

    if not S:
        S = []

    for m in pf['module']:
        if m in S:
            i = S.index(m)
            error("Circular dependency: "+ " -> ".join(S[i:] + [m]))
        S.append(m)

    for m in collect_use_deps(parsed_files, fn, src_dir):
        if m in mod2fn.keys():
            find_cycles(parsed_files, mod2fn, mod2fn[m], src_dir, S)

    for m in pf['module']:
        S.pop()

    pf['visited'] = True


# ============================================================================
def collect_pkg_deps(packages, p, archives=None, S=None):
    if not archives:
        archives = []

    if not S:
        S = []

    a = packages[p]['archive']
    if a in archives:
        return archives

    if a in S:
        i = S.index(a)
        error("Circular package dependency: "+ " -> ".join(S[i:] + [a]))
    S.append(a)

    for r in packages[p]['requires']:
        d = normpath(path.join(p, r))
        collect_pkg_deps(packages, d, archives, S)

    S.pop()

    if packages[p]['objects']:
        archives.insert(0, packages[p]['archive'])

    return archives


# ============================================================================
def error(msg):
    sys.stderr.write("makedep error: %s\n"%msg)
    sys.exit(1)

# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 8:
        print("Usage: {} <outfile> <project_name> <format> <mode> <archive.ext> <src-dir> <src-file1> [<src-file2> ...]"
              .format(sys.argv[0]))
        sys.exit(1)

    main(out_fn=sys.argv[1], project_name=sys.argv[2],
         mod_format=sys.argv[3], mode=sys.argv[4], archive_ext=sys.argv[5],
         src_dir=sys.argv[6], src_files=sys.argv[7:])
