#!/usr/bin/env python3
"""Drop the page cache of the model shards (POSIX_FADV_DONTNEED per file).

Targeted counterpart of the stage's own page-out on close, for the case where
a stage process exits without running its destructor (SIGTERM from systemd).
Only the named GGUF shards are affected; nothing else is dropped. Run only
when no process maps the shards (units stopped), otherwise mapped pages stay."""
import glob, os, sys

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: release-model-cache.py FIRST-SHARD.gguf")
    first = sys.argv[1]
    pattern = first.replace("-00001-of-", "-*-of-") if "-00001-of-" in first else first
    paths = sorted(glob.glob(pattern)) or [first]
    total = 0
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            total += size
        finally:
            os.close(fd)
        print(f"released page cache of {path} ({size/2**30:.3f} GiB)")
    print(f"total {total/2**30:.3f} GiB advised")

if __name__ == "__main__":
    main()
