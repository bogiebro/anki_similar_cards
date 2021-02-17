import numpy as np
import scipy.sparse as sp
from anki import hooks
from aqt import gui_hooks, mw, dialogs
from aqt.qt import *
from lxml.html import fromstring
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

def field_text(flds):
    for fld in flds:
        yield fromstring(fld).text_content() if fld else ""

def init_counts():
    global count_extractor, tfidf, ids, counts, vecs
    id_list = []
    def note_iterator():
        for id, flds in mw.col.db.execute("select id, flds from notes order by id"):
            if flds:
                id_list.append(id)
                yield " ".join(field_text(flds.split(chr(0x1f))))
    counts = count_extractor.transform(note_iterator())
    ids = np.array(id_list, dtype=np.long)
    vecs = tfidf.fit_transform(counts)

def handle_open_window():
    global suggestion_window
    suggestion_window.show()

action = QAction("Show Similar Notes", mw)
action.triggered.connect(handle_open_window)
mw.form.menuTools.addAction(action)

class MatchItem(QWidget):
    def __init__(self, note_id, itr):
        super().__init__()
        self.search_str = "nid:" + str(note_id)
        vbox = QVBoxLayout()
        self.setLayout(vbox)
        for ix, a in enumerate(filter(None, itr)):
            if len(a) > 120:
                a = a[:120] + "..."
            label = QLabel(a)
            vbox.addWidget(label)
            if ix > 0:
                label.setIndent(50)

    def handle_click(self):
        browser = dialogs.open("Browser", mw)
        browser.form.searchEdit.lineEdit().setText(self.search_str)
        browser.onSearchActivated()

class SuggestionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)
        font = QFont("Arial", 20)
        self.hint = QLabel("Open the note editor to show similar notes")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setFont(font)
        self.vbox.addWidget(self.hint)
        self.list_widget = None

    def handle_click(self, item):
        self.list_widget.itemWidget(item).handle_click()

    def clear(self):
        if self.list_widget is not None:
            self.list_widget.clear()
        if self.hint is not None:
            self.list_widget = QListWidget()
            self.list_widget.setAlternatingRowColors(True)
            self.list_widget.itemClicked.connect(self.handle_click)
            self.hint.setParent(None)
            self.hint = None
            self.vbox.addWidget(self.list_widget)

    def addItem(self, note_id, item_text):
        item = MatchItem(note_id, item_text)
        list_item = QListWidgetItem(self.list_widget)
        list_item.setSizeHint(item.sizeHint())
        self.list_widget.addItem(list_item)
        self.list_widget.setItemWidget(list_item, item)

def handle_modified_note(note, query_counts):
    global counts, vecs, ids
    ix = np.searchsorted(ids, note.id)
    if ix >= ids.shape[0] or ids[ix] != note.id: # New note
        ids = np.concatenate((ids[:ix], [note.id], ids[ix:]))
        counts = sp.vstack((counts[:ix,:], query_counts, counts[ix:,:]))
    else: # Old note
        counts = sp.vstack((counts[:ix,:], query_counts, counts[ix+1:,:]))
    vecs = tfidf.fit_transform(counts)

typing_cache = None
def handle_typing_timer(note):
    global typing_cache, suggestion_window, counts, ids
    text = " ".join(field_text(note.fields))
    text_hash = hash(text)
    if text_hash == typing_cache: return
    else:
        typing_cache = text_hash
        query_counts = count_extractor.transform([" ".join(note.fields)])
        query = tfidf.transform(query_counts)
        dot_prods = -(vecs @ query.T).A[:,0]
        mask = dot_prods < -0.14
        dot_prods = dot_prods[mask]
        if dot_prods.shape[0] > 8:
            max_ixs = np.argpartition(dot_prods, 8)[:9]
            high_dot_prods = dot_prods[max_ixs]
            sorted_ixs = np.argsort(high_dot_prods)
            matching_ids = ids[mask][max_ixs[sorted_ixs][1:]]
            # matching_prods = -high_dot_prods[sorted_ixs]
        else:
            sorted_ixs = np.argsort(dot_prods)
            matching_ids = ids[mask][sorted_ixs[1:]]
            # matching_prods = -dot_prods[sorted_ixs]
        suggestion_window.clear()
        for id in matching_ids:
            flds = mw.col.db.scalar(f"select flds from notes where id = {id}")
            if flds:
                suggestion_window.addItem(id, field_text(flds.split(chr(0x1f))))
        if note.id > 0:
            handle_modified_note(note, query)

gui_hooks.editor_did_fire_typing_timer.append(handle_typing_timer)
gui_hooks.editor_did_load_note.append(lambda editor: handle_typing_timer(editor.note))

def handle_deleted(_, note_ids):
    global ids, counts, vecs
    for id in note_ids:
        ix = np.searchsorted(ids, id)
        counts = sp.vstack((counts[:ix,:], counts[ix+1:,:]))
        ids = np.concatenate((ids[:ix], ids[ix+1:]))
        vecs = tfidf.fit_transform(counts)
hooks.notes_will_be_deleted.append(handle_deleted)

def init_hook():
    global count_extractor, tfidf, list_widget, suggestion_window
    suggestion_window = SuggestionWindow()
    count_extractor = HashingVectorizer(
        stop_words='english', alternate_sign=False, norm=None, ngram_range=(1,3))
    tfidf = TfidfTransformer()
    init_counts()

gui_hooks.main_window_did_init.append(init_hook)
