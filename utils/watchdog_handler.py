# src/watchdog_handler.py

import time
import os
import logging
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rag_helper import load_notes, split_notes, create_vector_store


def setup_logger(log_path=None):
    logger = logging.getLogger("WatchdogHandler")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


class NotesFileHandler(FileSystemEventHandler):
    def __init__(self, notes_path, logger):
        super().__init__()
        self.notes_path = os.path.abspath(notes_path)
        self.logger = logger
        self._debounce_time = 5  # seconds
        self._last_modified = 0

    def on_modified(self, event):
        if (
            not event.is_directory
            and os.path.abspath(event.src_path) == self.notes_path
        ):
            now = time.time()
            if now - self._last_modified < self._debounce_time:
                return  # Debounce rapid changes
            self._last_modified = now

            self.logger.info(
                f"Detected change in notes file: {self.notes_path}. Rebuilding FAISS index..."
            )
            try:
                docs = load_notes()
                chunks = split_notes(docs)
                create_vector_store(chunks)
                self.logger.info("FAISS vector store successfully updated.")
            except Exception as e:
                self.logger.error(
                    f"Error rebuilding FAISS vector store: {e}", exc_info=True
                )


def main():
    parser = argparse.ArgumentParser(
        description="Watch notes file and auto-rebuild FAISS vector store."
    )
    parser.add_argument(
        "--notes", type=str, default="data/notes.txt", help="Path to notes file"
    )
    parser.add_argument("--log", type=str, default=None, help="Optional log file path")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    notes_path = os.path.abspath(args.notes)
    watch_dir = os.path.dirname(notes_path) or "."

    logger.info(f"Starting watchdog on directory: {watch_dir} for file: {notes_path}")
    event_handler = NotesFileHandler(notes_path, logger)
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watchdog.")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
