from aqt import gui_hooks, mw
from anki import hooks
from itertools import chain
from aqt.qt import *
from sklearn.feature_extraction.text import HashingVectorizer
import os
import numpy as np
import scipy.sparse as sp
from lxml.html import fromstring

# TODO:
# Handle MathJax (requires re-doing view as html)
# Make a default label for the label if no note is selected for query
# Don't regenerate entire matrix on every modification
# Use idf, or better yet LDA
# Use BK-tree instead of exhaustive enumeration (profile this)
# Jump to note in browser on click
# Handle note syncing
# Re-write index on quit
# Remove 0 counts on save
# Don't show any of the top if the they're all below a threshold

def field_text(flds):
    for fld in flds:
        yield fromstring(fld).text_content() if fld else ""
    
def init_counts_file():
    global count_extractor
    ids = []
    def note_iterator():
        for id, flds in mw.col.db.execute("select id, flds from notes order by id"):
            ids.append(id)
            yield " ".join(field_text(flds.split(chr(0x1f))))
    ids = np.array(ids, dtype=np.long)
    return ids, count_extractor.transform(note_iterator())

def handle_open_window():
    global list_widget
    list_widget.show()

action = QAction("Show Similar Cards", mw)
action.triggered.connect(handle_open_window)
mw.form.menuTools.addAction(action)

class MatchItem(QWidget):
    def __init__(self, itr):
        super().__init__()
        vbox = QVBoxLayout()
        self.setLayout(vbox)
        for ix, a in enumerate(filter(None, itr)):
            if len(a) > 120:
                a = a[:120] + "..."
            label = QLabel(a)
            vbox.addWidget(label)
            if ix > 0:
                label.setIndent(50)

def handle_modified_note(note, query):
    global dirty_counts, counts, ids
    ix = np.searchsorted(ids, note.id)
    lil_counts = counts.tolil()
    lil_counts[ix,:] = query
    counts = lil_counts.tocsr()
    dirty_counts = True

typing_cache = None
def handle_typing_timer(note):
    global typing_cache, list_widget, counts, ids
    text = " ".join(field_text(note.fields))
    text_hash = hash(text)
    if text_hash == typing_cache: return
    else:
        typing_cache = text_hash
        query = count_extractor.transform([" ".join(note.fields)])
        dot_prods = (counts @ query.T).A[:,0]
        ixs = ids[np.argpartition(-dot_prods, 5)[:5]]
        list_widget.clear()
        for id, flds in mw.col.db.execute(
            f"select id, flds from notes where id in ({', '.join(map(str, ixs))})"):
            item = MatchItem(field_text(flds.split(chr(0x1f))))
            list_item = QListWidgetItem(list_widget)
            list_item.setSizeHint(item.sizeHint())
            list_widget.addItem(list_item)
            list_widget.setItemWidget(list_item, item)
            # eventually keep the ids so we can jump to them
        if note.id > 0:
            handle_modified_note(note, query)

gui_hooks.editor_did_fire_typing_timer.append(handle_typing_timer)
gui_hooks.editor_did_load_note.append(lambda editor: handle_typing_timer(editor.note))

def handle_deleted(_, note_ids):
    global dirty_counts, ids
    for id in note_ids:
        ix = np.searchsorted(ids, note.id)
        counts[ix,:] = 0
    dirty_counts = True
hooks.notes_will_be_deleted.append(handle_deleted)

def handle_exit():
    print("EXITING")
gui_hooks.profile_will_close.append(handle_exit)

def init_hook():
    global count_extractor, list_widget, counts, ids, dirty_counts
    dirty_counts = False

    list_widget = QListWidget()
    list_widget.setAlternatingRowColors(True)

    count_extractor = HashingVectorizer(stop_words='english', alternate_sign=False)

    # Initialize the wordcount database if it doesn't exist.
    PATH_THIS_ADDON = os.path.join(mw.pm.addonFolder(), __name__.split(".")[0])
    PATH_USER_DIR = os.path.join(PATH_THIS_ADDON, "user_files")
    os.makedirs(PATH_USER_DIR, exist_ok=True)
    COUNTS_FILE = os.path.join(PATH_USER_DIR, "counts.npz")
    IDS_FILE = os.path.join(PATH_USER_DIR, "ids.npy")
    if os.path.exists(COUNTS_FILE) and os.path.exists(IDS_FILE):
        counts = sp.load_npz(COUNTS_FILE)
        ids = np.load(IDS_FILE, allow_pickle=False)
    else:
        ids, counts = init_counts_file()
        sp.save_npz(COUNTS_FILE, counts)
        np.save(IDS_FILE, ids, allow_pickle=False)
        print("Initialized new counts file")

gui_hooks.main_window_did_init.append(init_hook)
