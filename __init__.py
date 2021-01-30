import os
from itertools import chain

import numpy as np
import scipy.sparse as sp
from anki import hooks
from aqt import gui_hooks, mw
from aqt.qt import *
from lxml.html import fromstring
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

# TODO: - Handle MathJax (requires re-doing view as html) - Make a default label
# for the label if no note is selected for query - Handle note syncing
# - When we show matching, we can use search "nid:1611866090425" in the browser.
# Jump to note in browser on click

def field_text(flds):
    for fld in flds:
        yield fromstring(fld).text_content() if fld else ""

def init_counts_file():
    global count_extractor, tfidf, ids
    id_list = []
    def note_iterator():
        for id, flds in mw.col.db.execute("select id, flds from notes order by id"):
            id_list.append(id)
            yield " ".join(field_text(flds.split(chr(0x1f))))
    counts = count_extractor.transform(note_iterator())
    ids = np.array(id_list, dtype=np.long)
    vecs = tfidf.fit_transform(counts)
    return ids, counts, vecs

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

def handle_modified_note(note, query_counts):
    global dirty_counts, counts, vecs, ids
    ix = np.searchsorted(ids, note.id)
    counts = sp.vstack((counts[:ix,:], query_counts, counts[ix+1:,:]))
    vecs = tfidf.fit_transform(counts)
    dirty_counts = True

typing_cache = None
def handle_typing_timer(note):
    global typing_cache, list_widget, counts, ids
    text = " ".join(field_text(note.fields))
    text_hash = hash(text)
    if text_hash == typing_cache: return
    else:
        typing_cache = text_hash
        query_counts = count_extractor.transform([" ".join(note.fields)])
        query = tfidf.transform(query_counts)
        dot_prods = (vecs @ query.T).A[:,0]
        max_ixs = np.argpartition(-dot_prods, 5)[:9]
        matching_ids = ids[max_ixs[dot_prods[max_ixs] > 0.1]]
        matching_ids = matching_ids[matching_ids != note.id]
        list_widget.clear()
        for id, flds in mw.col.db.execute(
            f"select id, flds from notes where id in ({', '.join(map(str, matching_ids))})"):
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
    global dirty_counts, ids, counts, vecs
    for id in note_ids:
        ix = np.searchsorted(ids, note.id)
        counts = sp.vstack((counts[:ix,:], counts[ix+1:,:]))
        vecs = tfidf.fit_transform(counts)
    dirty_counts = True
hooks.notes_will_be_deleted.append(handle_deleted)

def save_computed():
    global dirty_counts, counts, vecs, ids, COUNTS_FILE, VECS_FILE, IDS_FILE
    if dirty_counts:
        sp.save_npz(COUNTS_FILE, counts)
        sp.save_npz(VECS_FILE, vecs)
        np.save(IDS_FILE, ids, allow_pickle=False)
    dirty_counts = False

gui_hooks.profile_will_close.append(save_computed)

def load_db():
    "Initialize the wordcount database if it doesn't exist."
    global counts, vecs, ids, COUNTS_FILE, VECS_FILE, IDS_FILE
    PATH_THIS_ADDON = os.path.join(mw.pm.addonFolder(), __name__.split(".")[0])
    PATH_USER_DIR = os.path.join(PATH_THIS_ADDON, "user_files")
    os.makedirs(PATH_USER_DIR, exist_ok=True)
    COUNTS_FILE = os.path.join(PATH_USER_DIR, "counts.npz")
    IDS_FILE = os.path.join(PATH_USER_DIR, "ids.npy")
    VECS_FILE = os.path.join(PATH_USER_DIR, "vecs.npz")
    if os.path.exists(COUNTS_FILE) and os.path.exists(IDS_FILE) and os.path.exists(VECS_FILE):
        counts = sp.load_npz(COUNTS_FILE)
        vecs = sp.load_npz(VECS_FILE)
        ids = np.load(IDS_FILE, allow_pickle=False)
    else:
        ids, counts, vecs = init_counts_file()
        dirty_counts = True
        save_computed()
        print("Initialized new counts file")

def init_hook():
    global count_extractor, tfidf, list_widget, dirty_counts
    dirty_counts = False
    list_widget = QListWidget()
    list_widget.setAlternatingRowColors(True)
    count_extractor = HashingVectorizer(
        stop_words='english', alternate_sign=False, norm=None)
    tfidf = TfidfTransformer()
    load_db()

gui_hooks.main_window_did_init.append(init_hook)
