from cssd_ibscabc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-splits", *__import__("sys").argv[1:]]))
